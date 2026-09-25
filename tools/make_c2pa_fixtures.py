"""One-shot dev tool: signs the two positive-path C2PA test fixtures.

IMPORTANT: the signing key and certificate are THROWAWAY test credentials,
generated fresh on every run inside a temp dir and never written to the
repo. Only the signed IMAGES (which embed the public cert) are committed.
Real trust anchors arrive in Phase 8; until then verify.py keeps the trust
check OFF, which is why these self-signed fixtures validate as "Valid".

What it makes (in tests/fixtures/):
  c2pa_valid_camera.jpg  from real_01.jpg, declares camera capture
  c2pa_valid_ai.png      from synthetic_01.png, declares AI generation

Usage (inside .venv):
  python tools/make_c2pa_fixtures.py
"""

import datetime
import pathlib
import tempfile

import c2pa
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

# Where the input images live and where signed copies go.
FIXTURES = pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures"

# The two IPTC source-type markers used by the C2PA ecosystem.
CAPTURED_MARKER = "http://cv.iptc.org/newscodes/c2pa/captured"
AI_MARKER = "http://cv.iptc.org/newscodes/c2pa/trainedAlgorithmicMedia"


# Makes a throwaway self-signed ES256 cert that satisfies the c2pa-rs
# certificate profile (needs KeyUsage digitalSignature at minimum).
def make_throwaway_cert(key_path, cert_path):
    """Generate key + self-signed cert, write both as PEM files."""
    # Step 1: fresh random P-256 private key.
    key = ec.generate_private_key(ec.SECP256R1())

    # Step 2: self-signed identity (subject == issuer, nobody vouches).
    name = x509.Name(
        [
            x509.NameAttribute(
                NameOID.COMMON_NAME,
                "deep-test-fixture (THROWAWAY, untrusted)",
            )
        ]
    )

    # Step 3: 10-year validity window starting now.
    now = datetime.datetime.now(datetime.timezone.utc)
    ten_years = datetime.timedelta(days=3650)

    # Step 4: build the certificate with the extensions c2pa-rs wants.
    builder = x509.CertificateBuilder()
    builder = builder.subject_name(name)
    builder = builder.issuer_name(name)
    builder = builder.public_key(key.public_key())
    builder = builder.serial_number(x509.random_serial_number())
    builder = builder.not_valid_before(now)
    builder = builder.not_valid_after(now + ten_years)
    builder = builder.add_extension(
        x509.BasicConstraints(ca=False, path_length=None),
        True,
    )
    builder = builder.add_extension(
        x509.KeyUsage(
            digital_signature=True,
            content_commitment=False,
            key_encipherment=False,
            data_encipherment=False,
            key_agreement=False,
            key_cert_sign=False,
            crl_sign=False,
            encipher_only=False,
            decipher_only=False,
        ),
        True,
    )
    builder = builder.add_extension(
        x509.SubjectKeyIdentifier.from_public_key(key.public_key()),
        False,
    )
    builder = builder.add_extension(
        x509.AuthorityKeyIdentifier.from_issuer_public_key(key.public_key()),
        False,
    )
    builder = builder.add_extension(
        x509.ExtendedKeyUsage([ExtendedKeyUsageOID.EMAIL_PROTECTION]),
        False,
    )
    cert = builder.sign(key, hashes.SHA256())

    # Step 5: save both as PEM files.
    key_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_path.write_bytes(key_pem)
    cert_path.write_bytes(cert_pem)


# Signs one image with one action assertion.
def sign_image(src_name, dst_name, mime, source_type, key_pem, cert_pem):
    """Embed a C2PA manifest into a copy of src_name, save as dst_name."""
    # Step 1: build the signer from our throwaway credentials.
    info = c2pa.C2paSignerInfo(
        alg="es256",
        sign_cert=cert_pem,
        private_key=key_pem,
        ta_url=None,
    )
    signer = c2pa.Signer.from_info(info)

    # Step 2: minimal manifest skeleton.
    manifest = {
        "claim_generator": "deep-test-fixtures/0.1",
        "format": mime,
        "title": dst_name,
    }
    builder = c2pa.Builder(manifest)

    # Step 3: the single statement this fixture exists to carry.
    action = {
        "action": "c2pa.created",
        "digitalSourceType": source_type,
    }
    builder.add_action(action)

    # Step 4: sign source bytes into the destination file.
    src_path = FIXTURES / src_name
    dst_path = FIXTURES / dst_name
    with open(src_path, "rb") as src_file:
        with open(dst_path, "wb+") as dst_file:
            builder.sign(signer, mime, src_file, dst_file)

    # Step 5: read back and confirm c2pa accepts it.
    reader = c2pa.Reader.try_create(str(dst_path))
    state = reader.get_validation_state()
    reader.close()
    print(dst_name, "-> validation_state:", state)
    if state != "Valid":
        raise SystemExit(f"fixture {dst_name} did not validate: {state}")


# Signs both fixtures with a fresh throwaway keypair.
def main():
    """Generate throwaway creds in tempdir, sign 2 fixtures, print proof."""
    with tempfile.TemporaryDirectory() as tmp:
        # Step 1: throwaway credentials live ONLY in this temp dir.
        tmp_path = pathlib.Path(tmp)
        key_path = tmp_path / "key.pem"
        cert_path = tmp_path / "cert.pem"
        make_throwaway_cert(key_path, cert_path)
        key_pem = key_path.read_bytes()
        cert_pem = cert_path.read_bytes()

        # Step 2: camera fixture (declares captured).
        sign_image(
            "real_01.jpg",
            "c2pa_valid_camera.jpg",
            "image/jpeg",
            CAPTURED_MARKER,
            key_pem,
            cert_pem,
        )

        # Step 3: AI fixture (declares trained algorithmic media).
        sign_image(
            "synthetic_01.png",
            "c2pa_valid_ai.png",
            "image/png",
            AI_MARKER,
            key_pem,
            cert_pem,
        )

    # Step 4: temp dir (with key) is deleted here automatically.
    print("done: key material destroyed, signed fixtures in tests/fixtures/")


if __name__ == "__main__":
    main()

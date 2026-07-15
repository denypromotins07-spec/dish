#!/usr/bin/env python3
"""
Stage 30: Environment Encryptor for API Keys
Encrypts .env file using hardware-bound key (machine ID) for secure storage.
Secrets are only decrypted in locked RAM during runtime.
"""

import base64
import hashlib
import os
import subprocess
import sys
from pathlib import Path

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
except ImportError:
    print("[!] Installing cryptography package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "cryptography", "-q"])
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


def get_machine_id() -> bytes:
    """Get a unique hardware-bound identifier."""
    # Try multiple methods for cross-platform compatibility
    methods = [
        # Linux: machine-id
        lambda: Path("/etc/machine-id").read_text().strip().encode(),
        # Linux: dmidecode (requires root)
        lambda: subprocess.check_output(["dmidecode", "-s", "system-uuid"]).strip().encode(),
        # Cross-platform: CPU ID via lscpu
        lambda: subprocess.check_output(["lscpu", "-p=SERIAL"]).strip().encode(),
        # Fallback: hostname + MAC address
        lambda: f"{os.uname().nodename}:{':'.join(['%02x' % b for b in uuid.getnode().to_bytes(6, 'big')])}".encode(),
    ]
    
    import uuid
    
    for method in methods:
        try:
            result = method()
            if result:
                return hashlib.sha256(result).digest()
        except (FileNotFoundError, subprocess.CalledProcessError, PermissionError):
            continue
    
    # Ultimate fallback
    raise RuntimeError("Could not determine unique machine identifier")


def derive_key(machine_id: bytes, salt: bytes) -> bytes:
    """Derive encryption key from machine ID using PBKDF2."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100_000,
    )
    return base64.urlsafe_b64encode(kdf.derive(machine_id))


def encrypt_env(input_path: str, output_path: str) -> None:
    """Encrypt the .env file."""
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    if not input_file.exists():
        print(f"[ERROR] Input file not found: {input_file}")
        sys.exit(1)
    
    # Read plaintext env
    plaintext = input_file.read_bytes()
    
    # Get hardware-bound key
    print("[*] Deriving encryption key from hardware identifier...")
    machine_id = get_machine_id()
    
    # Generate random salt
    salt = os.urandom(16)
    
    # Derive key
    key = derive_key(machine_id, salt)
    fernet = Fernet(key)
    
    # Encrypt
    ciphertext = fernet.encrypt(plaintext)
    
    # Write salt + ciphertext
    output_file.write_bytes(salt + ciphertext)
    
    # Securely delete original if requested
    print(f"[+] Encrypted {input_file} -> {output_file}")
    print("[*] Original file should be securely deleted:")
    print(f"    shred -u {input_file}")


def decrypt_env(encrypted_path: str, output_path: str = None) -> bytes:
    """Decrypt the .env file into memory."""
    encrypted_file = Path(encrypted_path)
    
    if not encrypted_file.exists():
        print(f"[ERROR] Encrypted file not found: {encrypted_file}")
        sys.exit(1)
    
    # Read salt + ciphertext
    data = encrypted_file.read_bytes()
    salt = data[:16]
    ciphertext = data[16:]
    
    # Get hardware-bound key
    print("[*] Deriving decryption key from hardware identifier...")
    machine_id = get_machine_id()
    
    # Derive key
    key = derive_key(machine_id, salt)
    fernet = Fernet(key)
    
    # Decrypt
    try:
        plaintext = fernet.decrypt(ciphertext)
        if output_path:
            Path(output_path).write_bytes(plaintext)
            print(f"[+] Decrypted to {output_path}")
        return plaintext
    except Exception as e:
        print(f"[ERROR] Decryption failed. Wrong machine or corrupted file: {e}")
        sys.exit(1)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Encrypt/Decrypt .env files with hardware-bound keys")
    parser.add_argument("action", choices=["encrypt", "decrypt"], help="Action to perform")
    parser.add_argument("input", help="Input file path")
    parser.add_argument("-o", "--output", help="Output file path")
    
    args = parser.parse_args()
    
    if args.action == "encrypt":
        output = args.output or f"{args.input}.enc"
        encrypt_env(args.input, output)
    elif args.action == "decrypt":
        output = args.output or "/dev/shm/.env.decrypted"  # Default to RAM disk
        decrypt_env(args.input, output)


if __name__ == "__main__":
    main()

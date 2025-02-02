import os
from cryptography.fernet import Fernet, InvalidToken


class EncryptionService:
    def __init__(self, encryption_key: str):
        self.key = Fernet(encryption_key)

    def encrypt(self, data) -> str:
        encrypted_data = self.key.encrypt(data.encode()).decode()
        return encrypted_data

    def decrypt(self, encrypted_data) -> str:
        try:
            decrypted_data = self.key.decrypt(encrypted_data.encode()).decode()
            return decrypted_data
        except InvalidToken:
            raise ValueError("Invalid encryption token")

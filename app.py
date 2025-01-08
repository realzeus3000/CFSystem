import streamlit as st
import numpy as np
from PIL import Image
import io
from scipy.integrate import solve_ivp
import json
from datetime import datetime

class ChaoticSBoxEncryption:
    def __init__(self):
        self.a = 18.0
        self.b = 3.1
        self.c = 2.0
        self.d = 10.0
        self.e = 3.0
        self.f = 2.6
        self.g = 5.0
        self.h = 13.0
        self.initial_conditions = [0.2, 0.4, 0.6, 0.8]

    def chaotic_system(self, t, vars):
        x, y, z, w = vars
        dx_dt = self.a * y * z - self.b * x * z - self.c * w
        dy_dt = self.d * x - x * z - y
        dz_dt = self.e * x * y - self.f * z
        dw_dt = self.g * x * z + self.h * z * y
        return [dx_dt, dy_dt, dz_dt, dw_dt]

    def generate_sbox(self):
        t_span = (0, 50)
        t_eval = np.linspace(*t_span, 500000)

        try:
            solution = solve_ivp(
                self.chaotic_system,
                t_span,
                self.initial_conditions,
                t_eval=t_eval,
                method='RK45'
            )

            x = solution.y[0]
            normalized_x = ((x - np.min(x)) / (np.max(x) - np.min(x)) * 255).astype(np.uint8)

            unique_values = np.unique(normalized_x)

            if len(unique_values) < 256:
                missing_values = np.setdiff1d(np.arange(256), unique_values)
                sbox = np.concatenate([unique_values, missing_values])[:256]
                np.random.shuffle(sbox)
            else:
                sbox = unique_values[:256]
                np.random.shuffle(sbox)

            assert len(sbox) == 256, f"S-box must have 256 values, but has {len(sbox)}"
            assert len(np.unique(sbox)) == 256, "S-box must have 256 unique values"

            inv_sbox = np.zeros(256, dtype=np.uint8)
            for i, val in enumerate(sbox):
                inv_sbox[val] = i

            return sbox, inv_sbox

        except Exception as e:
            st.error(f"Error in generating S-box: {str(e)}")
            raise

    def encrypt_image(self, image, sbox):
        img_array = np.array(image)
        shape = img_array.shape

        if img_array.max() > 255 or img_array.min() < 0:
            raise ValueError("Image values must be between 0 and 255")

        flat_img = img_array.flatten()
        encrypted = np.array([sbox[p] for p in flat_img], dtype=np.uint8)
        return encrypted.reshape(shape)

    def decrypt_image(self, encrypted_image, inv_sbox):
        flat_img = encrypted_image.flatten()
        decrypted = np.array([inv_sbox[p] for p in flat_img], dtype=np.uint8)
        return decrypted.reshape(encrypted_image.shape)

    def generate_key_file(self, sbox, inv_sbox, fuzzy_key):
        key_data = {
            'a': self.a,
            'b': self.b,
            'c': self.c,
            'd': self.d,
            'e': self.e,
            'f': self.f,
            'g': self.g,
            'h': self.h,
            'initial_conditions': self.initial_conditions,
            'sbox': sbox.tolist(),
            'inv_sbox': inv_sbox.tolist(),
            'fuzzy_key': fuzzy_key,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        return json.dumps(key_data, indent=4)

    def load_key_file(self, key_data):
        try:
            data = json.loads(key_data)
            st.session_state.loaded_params = {
                'a': float(data['a']),
                'b': float(data['b']),
                'c': float(data['c']),
                'd': float(data['d']),
                'e': float(data['e']),
                'f': float(data['f']),
                'g': float(data['g']),
                'h': float(data['h']),
                'initial_conditions': [float(x) for x in data['initial_conditions']],
                'sbox': np.array(data['sbox'], dtype=np.uint8),
                'inv_sbox': np.array(data['inv_sbox'], dtype=np.uint8),
                'fuzzy_key': float(data['fuzzy_key'])
            }
            self.a = st.session_state.loaded_params['a']
            self.b = st.session_state.loaded_params['b']
            self.c = st.session_state.loaded_params['c']
            self.d = st.session_state.loaded_params['d']
            self.e = st.session_state.loaded_params['e']
            self.f = st.session_state.loaded_params['f']
            self.g = st.session_state.loaded_params['g']
            self.h = st.session_state.loaded_params['h']
            self.initial_conditions = st.session_state.loaded_params['initial_conditions']
        except Exception as e:
            st.error(f"Error loading key file: {str(e)}")
            raise

def fuzzy_differential_equation(key, size):
    sequence = []
    x = key
    for _ in range(size):
        x = 3.9 * x * (1 - x)
        sequence.append(int((x * 255) % 256))
    return sequence

def fuzzy_encrypt_image(image, key):
    img_array = np.array(image)
    rows, cols, channels = img_array.shape
    size = rows * cols * channels
    chaotic_sequence = fuzzy_differential_equation(key, size)
    chaotic_sequence = np.array(chaotic_sequence).reshape((rows, cols, channels))
    encrypted_array = np.bitwise_xor(img_array, chaotic_sequence)
    return Image.fromarray(encrypted_array.astype('uint8'))

def fuzzy_decrypt_image(encrypted_image, key):
    return fuzzy_encrypt_image(encrypted_image, key)

def main():
    st.set_page_config(page_title="Chaotic Image Encryption")
    st.title("Image Encryption using Chaotic S-box and Fuzzy Differential Equations")
    st.write("Upload an image and test the two-layer encryption/decryption process")

    if 'encryption' not in st.session_state:
        st.session_state.encryption = ChaoticSBoxEncryption()
    if 'encrypted_image' not in st.session_state:
        st.session_state.encrypted_image = None
    if 'key_file_content' not in st.session_state:
        st.session_state.key_file_content = None
    if 'loaded_params' not in st.session_state:
        st.session_state.loaded_params = None

    tab1, tab2 = st.tabs(["Encryption", "Decryption"])

    with tab1:
        st.header("Image Encryption")
        uploaded_file = st.file_uploader("Choose an image for encryption", type=['png', 'jpg', 'jpeg'],
                                         key='encrypt_upload')

        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            with col1:
                original_img = Image.open(uploaded_file).convert("RGB")
                st.image(original_img, caption="Original Image", use_container_width=True)

            st.subheader("System Parameters for Chaotic S-box")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.session_state.encryption.a = st.slider("Parameter a", 15.0, 21.0, 18.0, 0.1)
                st.session_state.encryption.b = st.slider("Parameter b", 2.0, 4.0, 3.1, 0.01)

            with col2:
                st.session_state.encryption.c = st.slider("Parameter c", 1.0, 3.0, 2.0, 0.01)
                st.session_state.encryption.d = st.slider("Parameter d", 8.0, 12.0, 10.0, 0.1)

            with col3:
                st.session_state.encryption.e = st.slider("Parameter e", 2.0, 4.0, 3.0, 0.01)
                st.session_state.encryption.f = st.slider("Parameter f", 2.0, 3.0, 2.6, 0.01)

            with col4:
                st.session_state.encryption.g = st.slider("Parameter g", 4.0, 6.0, 5.0, 0.01)
                st.session_state.encryption.h = st.slider("Parameter h", 12.0, 14.0, 13.0, 0.01)

            st.subheader("Initial Conditions for Chaotic S-box")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.session_state.encryption.initial_conditions[0] = st.number_input("x0", value=0.2, format="%.3f",
                                                                                    step=0.001)
            with col2:
                st.session_state.encryption.initial_conditions[1] = st.number_input("y0", value=0.4, format="%.3f",
                                                                                    step=0.001)
            with col3:
                st.session_state.encryption.initial_conditions[2] = st.number_input("z0", value=0.6, format="%.3f",
                                                                                    step=0.001)
            with col4:
                st.session_state.encryption.initial_conditions[3] = st.number_input("w0", value=0.8, format="%.3f",
                                                                                    step=0.001)

            fuzzy_key = st.number_input("Enter a key for Fuzzy Encryption (0 < key < 1):", min_value=0.001, max_value=0.999, value=0.5)

            if st.button("Encrypt Image", key="encrypt_btn"):
                with st.spinner("Encrypting..."):
                    try:
                        # First layer: Chaotic S-box encryption
                        sbox, inv_sbox = st.session_state.encryption.generate_sbox()
                        encrypted_1 = st.session_state.encryption.encrypt_image(original_img, sbox)

                        # Second layer: Fuzzy Differential Equation encryption
                        encrypted_2 = fuzzy_encrypt_image(Image.fromarray(encrypted_1), fuzzy_key)

                        buffer = io.BytesIO()
                        encrypted_2.save(buffer, format="PNG")
                        st.session_state.encrypted_image = buffer.getvalue()
                        st.session_state.key_file_content = st.session_state.encryption.generate_key_file(sbox, inv_sbox, fuzzy_key)

                        with col2:
                            st.image(encrypted_2, caption="Encrypted Image", use_container_width=True)

                    except Exception as e:
                        st.error(f"Encryption failed: {str(e)}")

            if st.session_state.encrypted_image is not None:
                download_col1, download_col2 = st.columns(2)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                with download_col1:
                    st.download_button(
                        label="Download Encrypted Image",
                        data=st.session_state.encrypted_image,
                        file_name=f"encrypted_image_{timestamp}.png",
                        mime="image/png"
                    )

                with download_col2:
                    st.download_button(
                        label="Download Encryption Key",
                        data=st.session_state.key_file_content,
                        file_name=f"encryption_key_{timestamp}.txt",
                        mime="text/plain"
                    )

                st.warning("Please save both the encrypted image and the key file. You will need both for decryption.")

    with tab2:
        st.header("Image Decryption")
        upload_col1, upload_col2 = st.columns(2)

        with upload_col1:
            encrypted_file = st.file_uploader("Choose an encrypted image", type=['png'], key='decrypt_upload')

        with upload_col2:
            key_file = st.file_uploader("Upload encryption key file", type=['txt'])

        if encrypted_file is not None and key_file is not None:
            try:
                key_data = key_file.read().decode('utf-8')
                st.session_state.encryption.load_key_file(key_data)

                col1, col2 = st.columns(2)
                with col1:
                    encrypted_img = Image.open(encrypted_file).convert("RGB")
                    st.image(encrypted_img, caption="Encrypted Image", use_container_width=True)

                st.subheader("Loaded Encryption Parameters")
                params_col1, params_col2, params_col3, params_col4 = st.columns(4)

                with params_col1:
                    st.text(f"a: {st.session_state.encryption.a:.3f}")
                    st.text(f"b: {st.session_state.encryption.b:.3f}")
                with params_col2:
                    st.text(f"c: {st.session_state.encryption.c:.3f}")
                    st.text(f"d: {st.session_state.encryption.d:.3f}")
                with params_col3:
                    st.text(f"e: {st.session_state.encryption.e:.3f}")
                    st.text(f"f: {st.session_state.encryption.f:.3f}")
                with params_col4:
                    st.text(f"g: {st.session_state.encryption.g:.3f}")
                    st.text(f"h: {st.session_state.encryption.h:.3f}")

                st.text("Initial Conditions:")
                st.write(f"[{', '.join([f'{x:.3f}' for x in st.session_state.encryption.initial_conditions])}]")

                fuzzy_key = st.session_state.loaded_params['fuzzy_key']

                st.text(f"Fuzzy Differential Key: {fuzzy_key:.6f}")

                if st.button("Decrypt Image", key="decrypt_btn"):
                    with st.spinner("Decrypting..."):
                        try:
                            inv_sbox = st.session_state.loaded_params['inv_sbox']

                            # First reverse Fuzzy Differential Equation encryption
                            decrypted_1 = fuzzy_decrypt_image(encrypted_img, fuzzy_key)

                            # Then reverse Chaotic S-box encryption
                            decrypted_2 = st.session_state.encryption.decrypt_image(np.array(decrypted_1), inv_sbox)
                            final_decrypted_image = Image.fromarray(decrypted_2)

                            with col2:
                                st.image(final_decrypted_image, caption="Decrypted Image", use_container_width=True)

                            buffer = io.BytesIO()
                            final_decrypted_image.save(buffer, format="PNG")
                            st.session_state.decrypted_image = buffer.getvalue()

                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            st.download_button(
                                label="Download Decrypted Image",
                                data=st.session_state.decrypted_image,
                                file_name=f"decrypted_image_{timestamp}.png",
                                mime="image/png"
                            )

                        except Exception as e:
                            st.error(f"Decryption failed: {str(e)}")

            except Exception as e:
                st.error(f"Error loading key file: {str(e)}")

if __name__ == "__main__":
    main()

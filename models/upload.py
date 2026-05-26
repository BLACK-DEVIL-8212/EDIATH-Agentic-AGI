from huggingface_hub import login, upload_file

login()

upload_file(
    path_or_fileobj="EDIATH-Q4_K_M.gguf",
    path_in_repo="EDIATH-Q4_K_M.gguf",
    repo_id="user052/EDIATH-Q4_K_M",
    repo_type="model",
)
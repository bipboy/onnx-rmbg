import tempfile


def create_tempfile_from_buffer(buffer):
    tmp_file = tempfile.NamedTemporaryFile(delete=False)
    tmp_file.write(buffer)
    tmp_file.close()
    return tmp_file.name

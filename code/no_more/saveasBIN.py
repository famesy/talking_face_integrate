newFileBytes = [123, 3, 255, 0, 100]
newFileByteArray = bytearray(newFileBytes)
newFile = open("filename.bin", "wb")
newFile.write(newFileByteArray)

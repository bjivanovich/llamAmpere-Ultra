import os

f_cmake = r"C:\Users\bj_iv\Desktop\llamAmpere-Ultra\src\CMakeLists.txt"
with open(f_cmake, 'r', encoding='utf-8') as fp:
    c = fp.read()

if "llama-ple-disk.cpp" not in c:
    c = c.replace(
        "    llama-model-loader.cpp\n",
        "    llama-model-loader.cpp\n    llama-ple-disk.cpp\n",
        1
    )
    with open(f_cmake, 'w', encoding='utf-8') as fp:
        fp.write(c)
    print("SUCCESS: added llama-ple-disk.cpp to src/CMakeLists.txt")
else:
    print("Already present")

import os
import sys

# Append the directory one level up to sys.path so we can import Deeploy
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Deeploy.TileLang.TextASTParser import TextASTParser
from Deeploy.DeeployTypes import NetworkContext
from collections import OrderedDict

def main():
    if len(sys.argv) < 2:
        print("Usage: python deeployRunner_tilelang_softhier.py <path_to_ast_txt>")
        sys.exit(1)
        
    ast_filepath = sys.argv[1]
    
    with open(ast_filepath, 'r') as f:
        ast_string = f.read()

    parser = TextASTParser()
    root = parser.parse(ast_string)

    if root is None:
        print("Failed to parse the AST string.")
        sys.exit(1)

    print("--- 1. Parsed AST String ---")
    ops = parser.walk_tree(root)
    import pprint
    pprint.pprint(ops)

    print("--- 2. Generating Deeploy ExecutionBlock (C Code) ---")
    eb = parser.generate_execution_block(ops)
    
    ctxt = NetworkContext(OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict())
    code = eb.generate(ctxt)
    
    out_file = "Network.c"
    with open(out_file, 'w') as f:
        f.write("#include \"flex_l1_malloc.h\"\n")
        f.write("#include \"flex_dma_async.h\"\n")
        f.write("#include \"flex_intra_cluster_sync.h\"\n")
        f.write("#include <stdint.h>\n")
        f.write("#include <string.h>\n\n")
        f.write("typedef _Float16 fp16;\n\n")
        f.write("void tilelang_main(fp16* A, fp16* B, fp16* C) {\n")
        f.write(code)
        f.write("\n}\n")
        
    print(f"--- 3. Wrote generated code to {out_file} ---")
    print("Ready to compile with GCC and simulate on SoftHier!")

if __name__ == "__main__":
    main()

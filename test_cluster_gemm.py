import tilelang
import tilelang.language as T

# This test defines a macro-kernel where two different clusters (Block IDs)
# perform a GEMM on the same tile of data sequentially.
# In a real scenario, this TIR will be fed to Deeploy's TilingExtension,
# which will break it down into L1-fitting micro-tiles.

@tilelang.jit
def test_cluster_gemm(A, B, M: int, N: int, K: int):
    # Two clusters will operate sequentially. 
    # Cluster 0 computes C = A @ B
    # Cluster 1 computes D = C @ B (or similar) on the same data.
    
    A: T.Tensor((M, K), T.float16)
    B: T.Tensor((K, N), T.float16)
    C = T.empty((M, N), T.float16)
    D = T.empty((M, N), T.float16)
    
    # We explicitly annotate the blocks to represent clusters.
    with T.Kernel(2, threads=1) as cluster_id:
        
        # Cluster 0: First GEMM
        if cluster_id == 0:
            for i, j, k in T.grid(M, N, K):
                with T.block("gemm0"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = T.float16(0)
                    C[vi, vj] += A[vi, vk] * B[vk, vj]
                    
        # Cluster 1: Second GEMM on the same tile (using C as input)
        with T.block("cluster1_guard"):
            if cluster_id == 1:
                for i, j, k in T.grid(M, N, K):
                    with T.block("gemm1"):
                        vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                        with T.init():
                            D[vi, vj] = T.float16(0)
                        D[vi, vj] += C[vi, vk] * B[vk, vj]

    return C, D

if __name__ == "__main__":
    M = 1024
    N = 1024
    K = 1024
    A = T.empty((M, K), T.float16)
    B = T.empty((K, N), T.float16)
    
    prim_func = test_cluster_gemm.get_tir(A, B, M=M, N=N, K=K)
    print("Generated TIR for macro-kernel:")
    print(prim_func)

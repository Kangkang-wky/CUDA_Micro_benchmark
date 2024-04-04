# lds & memory transaction & bank conflict 

## 简介

对于正常的 lds sts 指令, bank conflict 发生情况以及对 shared memory 广播机制, 在 guide programming book 中有详细的介绍与举例

但是对于 lds64, lds128 这种向量化访存指令, memory transaction 以及 shared memory 上的广播机制比较模糊,
通过参看博客中推荐的 micro benchmark 进行分析和测试

## shared memory 认识

放在 shared memory 中的数据是以 4 bytes (即 32 bits) 作为 1 个 word, 依次放在 32 个 banks 中。所以, 第 i 个 word, 就存放在第 ( i mod 32 ) 个 bank 上.

每个 bank 在每个 cycle 的 bandwidth 为 32 bits.

所以 shared memory 在每个 cycle 的 bandwidth 为 32 * 32 bits = 32 * 4 bytes = 128 bytes.

换句话说的话, 每次 memory transaction 最多访问 128 bytes 的数据.


## 非向量化访存时(这里假设是 4 bytes 一个 float 类型)

如果 warp 中每个 thread 只需要访问 4 bytes, 则 broadcast 和 bank conflicts 的机制很简单:

当多个 thread 访问同一个 bank 内的同一个 word, 就会触发 broadcast 机制. 这个 word 会同时发给对应的 thread:
当多个 thread 访问同一个 bank 内的不同 word 时, 就会产生 bank conflict. 于是请求会被拆分成多次 memory transaction, 串行地被发射 (issue) 出去执行. (比如 2-way bank conflict, 就拆分成 2 次 transaction)

换句话说, 

单次请求中，warp 内 32 个 thread，每个访问 4 bytes，那么总的数据需求就是最多 128 bytes。只要不产生 bank conflict，一次 memory transaction 就够了。取回来 128 bytes 的数据，warp 内怎么分都可以.

## 向量化访存时

### 64bit 访问

使用 LDS.64 指令(或者通过 float2、uint2 等类型) 取数据时，每个 thread 请求 64 bits (即 8 bytes) 数据，那么每 16 个 thread 就需要请求 128 bytes 的数据。

所以 CUDA 会默认将一个 warp 拆分为两个 half warp，每个 half warp 产生一次 memory transaction。即一共两次 transaction。

对于 64 位宽的访存指令而言，除非触发广播机制，否则一个 Warp 中有多少个活跃的 Half-Warp 就需要多少个 Memory Transaction，一个 Half-Warp 活跃的定义是这个 Half-Warp 内有任意一个线程活跃。触发广播机制只需满足以下条件中的至少一个：

对于 Warp 内所有活跃的第 i 号线程, 第 i xor 1 号线程不活跃或者访存地址和其一致;
(i.e. T0==T1, T2==T3, T4==T5, T6==T7, T8 == T9, ......, T30 == T31, etc.)

对于 Warp 内所有活跃的第 i 号线程, 第 i xor 2 号线程不活跃或者访存地址和其一致;
(i.e. T0==T2, T1==T3, T4==T6, T5==T7 etc.)

为什么呢?? 

简单理解一下，当上面两种情况发生时，硬件就可以判断(具体是硬件还是编译器的功劳，我也不确定，先归给硬件吧), 单个 half warp 内，最多需要 64 bytes 的数据，那么两个 half warp 就可以合并起来，通过一次 memory transaction，拿回 128 bytes 的数据. 然后线程之间怎么分都可以(broadcast 机制).

当然，这里的前提是没有产生 bank conflict。即没有从单个 bank 请求超过 1 个 word. 


注意！！

其实 bank conflict 是针对单次 memory transaction 而言的。如果单次 memory transaction 需要访问的 128 bytes 中有多个 word 属于同一个 bank，就产生了 bank conflict，从而需要拆分为多次 transaction。

比如这里，第一次访问了 0 - 31 个 word，第二次访问了 32 - 63 个 word，每次 transaction 内部并没有 bank conflict



micro_benchmark 测试, 具体可以看 png 中的 6 个 cases

### 128bit 访问

以下 quarter warp 的称呼并没有在 官方文档 中出现过.

使用 LDS.128 指令(或者通过 float4、uint4 等类型) 取数据时, 每个 thread 请求 128 bits(即 16 bytes) 数据，那么每 8 个 thread 就需要请求 128 bytes 的数据. 

所以，CUDA 会默认把每个 half warp 进一步切分成两个 quarter warp, 每个包含 8 个 thread. 每个 quarter warp 产生一次 memory transaction。所以每个 warp 每次请求，默认会有 4 次 memory transaction. （没有 bank conflict 的情况下).


类似 64 位宽的情况, 当满足特定条件时, 一个 half warp 内的两个 quarter warp 的访存请求会合并为 1 次 memory transaction. 但是两个 half warp 不会再进一步合并了.


对于 Warp 内所有活跃的第 i 号线程，第 i xor 1 号线程不活跃或者访存地址和其一致；(i.e. T0==T1, T2==T3, T4==T5, T6==T7, T8 == T9, ......, T30 == T31, etc.)
对于 Warp 内所有活跃的第 i 号线程，第 i xor 2 号线程不活跃或者访存地址和其一致；(i.e. T0==T2, T1==T3, T4==T6, T5==T7 etc.)
(活跃是指有访存需求)



## 参考

https://forums.developer.nvidia.com/t/how-to-understand-the-bank-conflict-of-shared-mem/260900

https://code.hitori.moe/post/cuda-shared-memory-access-mechanism-with-vectorized-instructions/

https://zhuanlan.zhihu.com/p/690052715

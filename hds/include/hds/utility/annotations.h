#pragma once

#if defined(__clang__)
/// Force function inline
#define FORCE_INLINE [[gnu::always_inline]]
#define NO_INLINE __attribute__((noinline))

#elif defined(__GNUC__) || defined(__GNUG__)
/// Force function inline
#define FORCE_INLINE [[gnu::always_inline]]
#define NO_INLINE __attribute__((noinline))
#endif

#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))

#define HDS_HOST_INLINE __host__ __forceinline__
#define HDS_DEVICE_INLINE __device__ __forceinline__
#define HDS_HOST_DEVICE_INLINE __host__ __device__ __forceinline__

#define HDS_HOST_NO_INLINE __host__ __noinline__
#define HDS_DEVICE_NO_INLINE __device__ __noinline__
#define HDS_HOST_DEVICE_NO_INLINE __host__ __device__ __noinline__

#define HOST __host__
#define DEVICE __device__
#define HOST_DEVICE HOST DEVICE

#else

#define HDS_HOST_INLINE FORCE_INLINE
#define HDS_DEVICE_INLINE FORCE_INLINE
#define HDS_HOST_DEVICE_INLINE FORCE_INLINE

#define HDS_HOST_NO_INLINE NO_INLINE
#define HDS_DEVICE_NO_INLINE NO_INLINE
#define HDS_HOST_DEVICE_NO_INLINE NO_INLINE

#define HOST 
#define DEVICE 
#define HOST_DEVICE 

#endif

#define HDS_HOST HDS_HOST_INLINE
#define HDS_DEVICE HDS_DEVICE_INLINE
#define HDS_HOST_DEVICE HDS_HOST_DEVICE_INLINE


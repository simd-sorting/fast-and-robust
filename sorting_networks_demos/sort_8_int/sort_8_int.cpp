#include <cstdio>
#include <immintrin.h>

/*
 * Bitonic sort of 8 elements with two vectors.
 * Each vector has a capacity of 4 elements.
 */

/* compute 4 COEX modules */
inline void COEX(__m128i &a, __m128i &b) {
  /* copy vector a to c */
  __m128i c = a;
  /* pairwise minimum */
  a = _mm_min_epi32(a, b);
  /* pairwise maximum */
  b = _mm_max_epi32(c, b);
}

/* shuffle 2 __m128i vectors */
#define SHUFFLE_TWO_VECS(a, b, mask)                                           \
  reinterpret_cast<__m128i>(_mm_shuffle_ps(                                    \
      reinterpret_cast<__m128>(a), reinterpret_cast<__m128>(b), mask));

int main() {
  int arr[8] = {11, 29, 13, 23, 37, 17, 19, 31};

  /* load vectors */
  __m128i v1 = _mm_loadu_si128(reinterpret_cast<__m128i *>(arr));
  __m128i v2 = _mm_loadu_si128(reinterpret_cast<__m128i *>(arr + 4));

  /* step 1 */
  COEX(v1, v2);
  v2 = _mm_shuffle_epi32(v2, _MM_SHUFFLE(2, 3, 0, 1));

  /* step 2 */
  COEX(v1, v2);
  auto tmp = v1;
  v1 = SHUFFLE_TWO_VECS(v1, v2, 0b10001000);
  v2 = SHUFFLE_TWO_VECS(tmp, v2, 0b11011101);

  /* step 3 */
  COEX(v1, v2);
  v2 = _mm_shuffle_epi32(v2, _MM_SHUFFLE(0, 1, 2, 3));

  /* step 4 */
  COEX(v1, v2);
  tmp = v1;
  v1 = SHUFFLE_TWO_VECS(v1, v2, 0b01000100);
  v2 = SHUFFLE_TWO_VECS(tmp, v2, 0b11101110);

  /* step 5 */
  COEX(v1, v2);
  tmp = v1;
  v1 = SHUFFLE_TWO_VECS(v1, v2, 0b10001000);
  v2 = SHUFFLE_TWO_VECS(tmp, v2, 0b11011101);

  /* step 6 */
  COEX(v1, v2);

  /* restore order */
  tmp = _mm_shuffle_epi32(v1, _MM_SHUFFLE(2, 3, 0, 1));
  auto tmp2 = _mm_shuffle_epi32(v2, _MM_SHUFFLE(2, 3, 0, 1));
  v2 = _mm_blend_epi32(tmp, v2, 0b00001010);
  v1 = _mm_blend_epi32(v1, tmp2, 0b00001010);

  /* save vectors */
  _mm_storeu_si128(reinterpret_cast<__m128i *>(arr), v1);
  _mm_storeu_si128(reinterpret_cast<__m128i *>(arr + 4), v2);

  for (int i = 0; i < 8; ++i)
    printf("%d ", arr[i]);
}

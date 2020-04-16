#include <cstdio>
#include <immintrin.h>

/*
 * Bitonic Sort of 16 elements with two vectors.
 * Each vector has a capacity of 8 elements.
 */

/* compute 8 COEX modules */
inline void COEX(__m256i &a, __m256i &b) {
  /* copy vector a to c */
  __m256i c = a;
  /* pairwise minimum */
  a = _mm256_min_epi32(a, b);
  /* pairwise maximum */
  b = _mm256_max_epi32(c, b);
}

/* shuffle 2 vectors, instruction for int is missing so use instruction for floats */
#define SHUFFLE_2_VECS(a, b, mask)                                             \
  reinterpret_cast<__m256i>(_mm256_shuffle_ps(                                 \
      reinterpret_cast<__m256>(a), reinterpret_cast<__m256>(b), mask));

/* bitonic sort for 16 integer */
inline void sort_16(__m256i &v1, __m256i &v2) {
  /* step 1 */
  COEX(v1, v2);

  /* step 2 */
  v2 = _mm256_shuffle_epi32(v2, _MM_SHUFFLE(2, 3, 0, 1));
  COEX(v1, v2);

  /* step 3 */
  auto tmp = v1;
  v1 = SHUFFLE_2_VECS(v1, v2, 0b10001000);
  v2 = SHUFFLE_2_VECS(tmp, v2, 0b11011101);
  COEX(v1, v2);

  /* step 4 */
  v2 = _mm256_shuffle_epi32(v2, _MM_SHUFFLE(0, 1, 2, 3));
  COEX(v1, v2);

  /* step 5 */
  tmp = v1;
  v1 = SHUFFLE_2_VECS(v1, v2, 0b01000100);
  v2 = SHUFFLE_2_VECS(tmp, v2, 0b11101110);
  COEX(v1, v2);

  /* step 6 */
  tmp = v1;
  v1 = SHUFFLE_2_VECS(v1, v2, 0b11011000);
  v2 = SHUFFLE_2_VECS(tmp, v2, 0b10001101);
  COEX(v1, v2);

  /* step 7 */
  v2 = _mm256_permutevar8x32_epi32(v2, _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0));
  COEX(v1, v2);

  /* step 8 */
  tmp = v1;
  v1 = SHUFFLE_2_VECS(v1, v2, 0b11011000);
  v2 = SHUFFLE_2_VECS(tmp, v2, 0b10001101);
  COEX(v1, v2);

  /* step 9 */
  tmp = v1;
  v1 = SHUFFLE_2_VECS(v1, v2, 0b11011000);
  v2 = SHUFFLE_2_VECS(tmp, v2, 0b10001101);
  COEX(v1, v2);

  /* permute to restore order more easily */
  v1 = _mm256_permutevar8x32_epi32(v1, _mm256_setr_epi32(0, 4, 1, 5, 6, 2, 7, 3));
  v2 = _mm256_permutevar8x32_epi32(v2, _mm256_setr_epi32(0,4,1,5,6,2,7,3));
  
  /* step 10 */
  tmp = v1;
  v1 = SHUFFLE_2_VECS(v1, v2, 0b10001000);
  v2 = SHUFFLE_2_VECS(tmp, v2, 0b11011101);
  COEX(v1, v2);

  /* restore order */
  auto b2 = _mm256_shuffle_epi32(v2,0b10110001);
  auto b1 = _mm256_shuffle_epi32(v1,0b10110001);
  v1 = _mm256_blend_epi32(v1, b2, 0b10101010);
  v2 = _mm256_blend_epi32(b1, v2, 0b10101010);
}

int main() {
  int arr[16] = {11, 3, 29, 7, 13, 23, 37, 1, 17, 8, 19, 31, 2, 33, 5, 14};

  /* load vectors */
  __m256i v1 = _mm256_loadu_si256(reinterpret_cast<__m256i *>(arr));
  __m256i v2 = _mm256_loadu_si256(reinterpret_cast<__m256i *>(arr + 8));

  /* sort */
  sort_16(v1, v2);

  /* save vectors */
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(arr), v1);
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(arr + 8), v2);

  for (int i = 0; i < 16; ++i)
    printf("%d ", arr[i]);
}

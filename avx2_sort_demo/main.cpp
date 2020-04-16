#include "avx2sort.h"
#include <functional>
#include <random>
#include <vector>

using namespace std;

/* this code demonstrates how to use avx2sort.h for sorting integers */

int main() {
  int n = 1000000;

  /* create a vector with random integers */
  vector<int> a(n);
  auto rand_int = bind(uniform_int_distribution<int>{INT32_MIN, INT32_MAX}, default_random_engine{std::random_device{}()});
  for (int i = 0; i < n; ++i) {
    a[i] = rand_int();
  }
  auto b = a;
  auto c = a;

  /* sort with avx2 */
  avx2::quicksort(a.data(), n);

  const int k = n / 2;
  /* instead of nth_element(begin(c), begin(c) + k, end(c)) */
  avx2::quickselect(b.data(), n, k);

  /* test if functions do what they should */
  sort(begin(b), begin(b) + k);
  sort(begin(b) + k + 1, end(b));
  sort(begin(c), end(c));
  if(a != c || b != c)
    puts("fail!!!");
  else
    puts("it works");
}

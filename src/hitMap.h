#ifndef HITMAP_H
#define HITMAP_H

#include <cstdint>
#include <cstring>

template <size_t kNumBits, size_t kNumHashes>
class hitMap {
public:
  hitMap();
  ~hitMap();

  void add(const char *key, const size_t keylen);
  bool contains(const uint32_t hash[kNumHashes]) const;
  void getHashes(uint32_t hash[kNumHashes]) const;

private:
  uint8_t *bits_;
};

#include "hitMap.cpp"

#endif // HITMAP_H


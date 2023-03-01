#ifndef MEMCACHED_WRAPPER_H
#define MEMCACHED_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

//assoc.c
void assoc_init(void);
item *assoc_find(const char *key, const size_t nkey, const uint32_t hv);
int assoc_insert(item *item, const uint32_t hv);
void assoc_delete(const char *key, const size_t nkey, const uint32_t hv);
void do_assoc_move_next_bucket(void);

//authfile.c
int load_user_db(void);
int check_user(const char *user, const char *password);

//base64.c
void base64_encode(const unsigned char *in, int in_len, unsigned char *out, int out_len);
int base64_decode(const char *in, int in_len, unsigned char *out, int out_len);



#ifdef __cplusplus
}
#endif

#endif /* MEMCACHED_WRAPPER_H */


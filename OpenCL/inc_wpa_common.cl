/**
 * Author......: See docs/credits.txt
 * License.....: MIT
 */

// m22000 and m22001 carry the same WPA record: the same ESSID, MACs, type, FT fields, nonces and
// EAPOL frame. The layout and the two helpers that read it live here so the two modes cannot drift
// apart.

// A WPA2 M2 frame can pass 256 bytes once the RSN extension element is present, so the limit on a
// stored EAPOL frame follows the frame rather than the old hccapx field it was inherited from.

#define WPA_EAPOL_LEN_MAX 512

typedef struct wpa
{
  u32  essid_buf[16];
  u32  essid_len;

  u32  mac_ap[2];
  u32  mac_sta[2];

  u32  type;            // 1 = PMKID, 2 = EAPOL, 3 = FT PMKID, 4 = FT EAPOL

  // FT specific

  u32  mdid;

  u32  r0khid[12];
  u32  r0khid_len;

  u32  r1khid[12];
  u32  r1khid_len;

  // Each KDF input is built on the host and sized to the longest ESSID and key holder ID the format
  // allows, not to a SHA-256 block, because the kernel hashes only the bytes it was given.

  u32  pke_r0[25];      // 19 + ESSID (32) + R0KH-ID (48) = 99 bytes
  u32  pke_r1[16];      // 15 + R1KH-ID (48)              = 63 bytes

  // PMKID specific

  u32  pmkid[4];
  u32  pmkid_data[19];  // 20 bytes for a PMKID, 28 + R1KH-ID (48) = 76 for an FT PMKID

  // EAPOL specific

  u32  keymic[4];
  u32  anonce[8];

  u32  keyver;

  u32  eapol[(WPA_EAPOL_LEN_MAX / 4) + 16];
  u32  eapol_len;

  u32  pke[32];

  int  message_pair_chgd;
  u32  message_pair;

  int  nonce_error_corrections_chgd;
  int  nonce_error_corrections;

  int  nonce_compare;
  int  detected_le;
  int  detected_be;

} wpa_t;

DECLSPEC void make_kn (PRIVATE_AS u32 *k)
{
  u32 kl[4];
  u32 kr[4];

  kl[0] = (k[0] << 1) & 0xfefefefe;
  kl[1] = (k[1] << 1) & 0xfefefefe;
  kl[2] = (k[2] << 1) & 0xfefefefe;
  kl[3] = (k[3] << 1) & 0xfefefefe;

  kr[0] = (k[0] >> 7) & 0x01010101;
  kr[1] = (k[1] >> 7) & 0x01010101;
  kr[2] = (k[2] >> 7) & 0x01010101;
  kr[3] = (k[3] >> 7) & 0x01010101;

  const u32 c = kr[0] & 1;

  kr[0] = kr[0] >> 8 | kr[1] << 24;
  kr[1] = kr[1] >> 8 | kr[2] << 24;
  kr[2] = kr[2] >> 8 | kr[3] << 24;
  kr[3] = kr[3] >> 8;

  k[0] = kl[0] | kr[0];
  k[1] = kl[1] | kr[1];
  k[2] = kl[2] | kr[2];
  k[3] = kl[3] | kr[3];

  k[3] ^= c * 0x87000000;
}

DECLSPEC void hmac_sha1_run_V (PRIVATE_AS u32x *w0, PRIVATE_AS u32x *w1, PRIVATE_AS u32x *w2, PRIVATE_AS u32x *w3, PRIVATE_AS const u32x *ipad, PRIVATE_AS const u32x *opad, PRIVATE_AS u32x *digest)
{
  // One PRF step. The message is the 20 byte digest and it is the only part of the padded block that
  // changes, between the two hashes and between steps, so the caller sets the rest once.

  w0[0] = digest[0];
  w0[1] = digest[1];
  w0[2] = digest[2];
  w0[3] = digest[3];
  w1[0] = digest[4];

  digest[0] = ipad[0];
  digest[1] = ipad[1];
  digest[2] = ipad[2];
  digest[3] = ipad[3];
  digest[4] = ipad[4];

  sha1_transform_vector (w0, w1, w2, w3, digest);

  w0[0] = digest[0];
  w0[1] = digest[1];
  w0[2] = digest[2];
  w0[3] = digest[3];
  w1[0] = digest[4];

  digest[0] = opad[0];
  digest[1] = opad[1];
  digest[2] = opad[2];
  digest[3] = opad[3];
  digest[4] = opad[4];

  sha1_transform_vector (w0, w1, w2, w3, digest);
}

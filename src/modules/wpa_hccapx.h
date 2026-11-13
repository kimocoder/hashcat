/**
 * Author......: See docs/credits.txt
 * License.....: MIT
 */

// The binary record the WPA modes read. 2500 and 2501 read it directly, and the hccapx import of
// 22000 and 22001 converts one record into the text form. The layout is the format hcxpcapngtool
// writes, so all four modes have to agree on it.

#ifndef WPA_HCCAPX_H
#define WPA_HCCAPX_H

#define HCCAPX_VERSION   4
#define HCCAPX_SIGNATURE 0x58504348 // HCPX

// this is required to force mingw to accept the packed attribute
#pragma pack(push,1)

struct auth_packet
{
  u8  version;
  u8  type;
  u16 length;
  u8  key_descriptor;
  u16 key_information;
  u16 key_length;
  u64 replay_counter;
  u8  wpa_key_nonce[32];
  u8  wpa_key_iv[16];
  u8  wpa_key_rsc[8];
  u8  wpa_key_id[8];
  u8  wpa_key_mic[16];
  u16 wpa_key_data_length;

} __attribute__((packed));

typedef struct auth_packet auth_packet_t;

struct hccapx
{
  u32 signature;
  u32 version;
  u8  message_pair;
  u8  essid_len;
  u8  essid[32];
  u8  keyver;
  u8  keymic[16];
  u8  mac_ap[6];
  u8  nonce_ap[32];
  u8  mac_sta[6];
  u8  nonce_sta[32];
  u16 eapol_len;
  u8  eapol[256];

} __attribute__((packed));

typedef struct hccapx hccapx_t;

#pragma pack(pop)

#endif
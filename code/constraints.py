"""constraints.py - Candidate registry of necessary flow-feature consistency conditions.

Every candidate is derived from a feature definition or from the documentation of the extraction
tool (CICFlowMeter for CIC-IDS2017; Argus/Bro-IDS feature table NUSW-NB15_features.csv for UNSW-NB15),
never from a relation that merely fits the data. A candidate enters the FROZEN audit list only if it
also passes the real-record conformance gate (see 03_real_record_audit.py and protocol/PROTOCOL.md).

A constraint is evaluated on a "view": dict feature-name -> numpy array (any common shape).
Categorical features are integer codes; `meta['codes'][feature][label]` gives the code of a label.
Each constraint returns (applicable, violated) boolean arrays; a record violates the constraint
when applicable & violated.

Classes
  D  domain/type      : sign, integrality, documented numeric range
  O  ordering         : min <= mean <= max, std <= max - min, bounded counts
  A  arithmetic       : identities that follow from the definition of a derived feature
  C  conditional-zero : a quantity that cannot exist when the packets it is computed from do not exist
  P  protocol         : protocol-dependent fields (UNSW-NB15 only; the CIC-IDS2017 CSVs carry no protocol column)
"""
import numpy as np

INT_TOL = 1e-6      # |x - round(x)| allowed for integer-valued features
ORD_TOL = 1e-3      # absolute slack for ordering relations (CSV precision)
EQ_ATOL = 1.0       # absolute slack for arithmetic identities on byte/packet totals
EQ_RTOL = 1e-3      # relative slack for arithmetic identities


class Constraint:
    def __init__(self, cid, cls, desc, source, feats, fn):
        self.id, self.cls, self.desc, self.source, self.feats, self.fn = cid, cls, desc, source, feats, fn

    def check(self, V, meta):
        app, viol = self.fn(V, meta)
        app = np.broadcast_to(np.asarray(app, dtype=bool), np.shape(V[self.feats[0]]))
        return app, np.asarray(viol, dtype=bool) & app


def _true(a):
    return np.ones(np.shape(a), dtype=bool)


def _neq(a, b, atol=EQ_ATOL, rtol=EQ_RTOL):
    return ~(np.abs(a - b) <= atol + rtol * np.abs(b))


# --------------------------------------------------------------------------------------------
# CIC-IDS2017 (CICFlowMeter features)
# --------------------------------------------------------------------------------------------
CIC_SRC = 'CICFlowMeter feature definitions (Lashkari et al., 2017; Sharafaldin et al., 2018)'

CIC_INT = ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Total Length of Fwd Packets',
           'Total Length of Bwd Packets', 'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Bwd Packet Length Max',
           'Bwd Packet Length Min', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Max', 'Fwd IAT Min',
           'Bwd IAT Total', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Fwd URG Flags', 'Fwd Header Length',
           'Bwd Header Length', 'Min Packet Length', 'Max Packet Length', 'FIN Flag Count', 'SYN Flag Count',
           'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count',
           'Down/Up Ratio', 'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
           'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward',
           'Active Max', 'Active Min', 'Idle Max', 'Idle Min']
CIC_SIGNED_OK = ['Init_Win_bytes_forward', 'Init_Win_bytes_backward']   # -1 encodes "no TCP window observed"
CIC_FLAGS = ['FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
             'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count']
CIC_GROUPS = {'FwdLen': 'Fwd Packet Length', 'BwdLen': 'Bwd Packet Length', 'FlowIAT': 'Flow IAT',
              'FwdIAT': 'Fwd IAT', 'BwdIAT': 'Bwd IAT', 'Active': 'Active', 'Idle': 'Idle'}
CIC_BWD_ZERO = ['Total Length of Bwd Packets', 'Bwd Packet Length Max', 'Bwd Packet Length Min',
                'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std',
                'Bwd IAT Max', 'Bwd IAT Min', 'Bwd Header Length', 'Bwd Packets/s', 'Avg Bwd Segment Size',
                'Subflow Bwd Packets', 'Subflow Bwd Bytes']
CIC_FWD_IAT = ['Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min']
CIC_BWD_IAT = ['Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min']


def cic_candidates(features):
    C = []
    add = lambda *a: C.append(Constraint(*a))
    F, B, DUR = 'Total Fwd Packets', 'Total Backward Packets', 'Flow Duration'
    for f in features:
        if f in CIC_SIGNED_OK:
            add(f'CIC-D-range[{f}]', 'D', f'{f} in [-1, 65535] (16-bit TCP window; -1 = not observed)', CIC_SRC, [f],
                lambda V, m, f=f: (_true(V[f]), (V[f] < -1 - INT_TOL) | (V[f] > 65535 + INT_TOL)))
        else:
            add(f'CIC-D-nonneg[{f}]', 'D', f'{f} >= 0 (counts, lengths, durations and rates cannot be negative)', CIC_SRC, [f],
                lambda V, m, f=f: (_true(V[f]), V[f] < -1e-9))
    for f in CIC_INT:
        if f in features:
            add(f'CIC-D-int[{f}]', 'D', f'{f} is integer-valued (packet/byte/flag count or microsecond tick)', CIC_SRC, [f],
                lambda V, m, f=f: (_true(V[f]), np.abs(V[f] - np.round(V[f])) > INT_TOL))
    add('CIC-D-minpkt', 'D', 'A flow record contains at least one forward packet', CIC_SRC, [F],
        lambda V, m: (_true(V[F]), V[F] < 1 - INT_TOL))
    # ordering inside each statistical group
    for g, p in CIC_GROUPS.items():
        mn, me, mx, sd = f'{p} Min', f'{p} Mean', f'{p} Max', f'{p} Std'
        add(f'CIC-O-minmean[{g}]', 'O', f'{mn} <= {me}', CIC_SRC, [mn, me], lambda V, m, a=mn, b=me: (_true(V[a]), V[a] > V[b] + ORD_TOL))
        add(f'CIC-O-meanmax[{g}]', 'O', f'{me} <= {mx}', CIC_SRC, [me, mx], lambda V, m, a=me, b=mx: (_true(V[a]), V[a] > V[b] + ORD_TOL))
        add(f'CIC-O-stdrange[{g}]', 'O', f'{sd} <= {mx} - {mn}', CIC_SRC, [sd, mx, mn],
            lambda V, m, s=sd, a=mx, b=mn: (_true(V[s]), V[s] > (V[a] - V[b]) + ORD_TOL))
    add('CIC-O-minmean[PktLen]', 'O', 'Min Packet Length <= Packet Length Mean', CIC_SRC, ['Min Packet Length', 'Packet Length Mean'],
        lambda V, m: (_true(V[F]), V['Min Packet Length'] > V['Packet Length Mean'] + ORD_TOL))
    add('CIC-O-meanmax[PktLen]', 'O', 'Packet Length Mean <= Max Packet Length', CIC_SRC, ['Packet Length Mean', 'Max Packet Length'],
        lambda V, m: (_true(V[F]), V['Packet Length Mean'] > V['Max Packet Length'] + ORD_TOL))
    add('CIC-O-stdrange[PktLen]', 'O', 'Packet Length Std <= Max Packet Length - Min Packet Length', CIC_SRC,
        ['Packet Length Std', 'Max Packet Length', 'Min Packet Length'],
        lambda V, m: (_true(V[F]), V['Packet Length Std'] > V['Max Packet Length'] - V['Min Packet Length'] + ORD_TOL))
    for f in CIC_FLAGS:
        add(f'CIC-O-flag<=pkts[{f}]', 'O', f'{f} <= total packets in the flow', CIC_SRC, [f, F, B],
            lambda V, m, f=f: (_true(V[f]), V[f] > V[F] + V[B] + INT_TOL))
    add('CIC-O-fwdpsh<=fwd', 'O', 'Fwd PSH Flags <= Total Fwd Packets', CIC_SRC, ['Fwd PSH Flags', F],
        lambda V, m: (_true(V[F]), V['Fwd PSH Flags'] > V[F] + INT_TOL))
    add('CIC-O-fwdurg<=fwd', 'O', 'Fwd URG Flags <= Total Fwd Packets', CIC_SRC, ['Fwd URG Flags', F],
        lambda V, m: (_true(V[F]), V['Fwd URG Flags'] > V[F] + INT_TOL))
    add('CIC-O-actdata<=fwd', 'O', 'act_data_pkt_fwd <= Total Fwd Packets', CIC_SRC, ['act_data_pkt_fwd', F],
        lambda V, m: (_true(V[F]), V['act_data_pkt_fwd'] > V[F] + INT_TOL))
    add('CIC-O-fwdhdr>=8n', 'O', 'Fwd Header Length >= 8 bytes x Total Fwd Packets (smallest transport header)', CIC_SRC, ['Fwd Header Length', F],
        lambda V, m: (_true(V[F]), V['Fwd Header Length'] < 8 * V[F] - INT_TOL))
    add('CIC-O-bwdhdr>=8n', 'O', 'Bwd Header Length >= 8 bytes x Total Backward Packets', CIC_SRC, ['Bwd Header Length', B],
        lambda V, m: (_true(V[B]), V['Bwd Header Length'] < 8 * V[B] - INT_TOL))
    add('CIC-O-subfwdpk<=fwd', 'O', 'Subflow Fwd Packets <= Total Fwd Packets', CIC_SRC, ['Subflow Fwd Packets', F],
        lambda V, m: (_true(V[F]), V['Subflow Fwd Packets'] > V[F] + INT_TOL))
    add('CIC-O-subbwdpk<=bwd', 'O', 'Subflow Bwd Packets <= Total Backward Packets', CIC_SRC, ['Subflow Bwd Packets', B],
        lambda V, m: (_true(V[B]), V['Subflow Bwd Packets'] > V[B] + INT_TOL))
    add('CIC-O-subfwdby<=tot', 'O', 'Subflow Fwd Bytes <= Total Length of Fwd Packets', CIC_SRC, ['Subflow Fwd Bytes', 'Total Length of Fwd Packets'],
        lambda V, m: (_true(V[F]), V['Subflow Fwd Bytes'] > V['Total Length of Fwd Packets'] + INT_TOL))
    add('CIC-O-subbwdby<=tot', 'O', 'Subflow Bwd Bytes <= Total Length of Bwd Packets', CIC_SRC, ['Subflow Bwd Bytes', 'Total Length of Bwd Packets'],
        lambda V, m: (_true(V[F]), V['Subflow Bwd Bytes'] > V['Total Length of Bwd Packets'] + INT_TOL))
    for d, n in (('Fwd', F), ('Bwd', B)):
        add(f'CIC-O-iattot<=dur[{d}]', 'O', f'{d} IAT Total <= Flow Duration', CIC_SRC, [f'{d} IAT Total', DUR],
            lambda V, m, d=d: (_true(V[DUR]), V[f'{d} IAT Total'] > V[DUR] + 1))
    add('CIC-O-flowiatmax<=dur', 'O', 'Flow IAT Max <= Flow Duration', CIC_SRC, ['Flow IAT Max', DUR],
        lambda V, m: (_true(V[DUR]), V['Flow IAT Max'] > V[DUR] + 1))
    add('CIC-O-actidle<=dur', 'O', 'Active Max and Idle Max <= Flow Duration', CIC_SRC, ['Active Max', 'Idle Max', DUR],
        lambda V, m: (_true(V[DUR]), (V['Active Max'] > V[DUR] + 1) | (V['Idle Max'] > V[DUR] + 1)))
    add('CIC-O-minpl<=fwdmin', 'O', 'Min Packet Length <= Fwd Packet Length Min', CIC_SRC, ['Min Packet Length', 'Fwd Packet Length Min'],
        lambda V, m: (_true(V[F]), V['Min Packet Length'] > V['Fwd Packet Length Min'] + INT_TOL))
    # arithmetic identities
    for d, n in (('Fwd', F), ('Bwd', B)):
        add(f'CIC-A-total=mean*n[{d}]', 'A', f'Total Length of {d} Packets = {d} Packet Length Mean x packet count', CIC_SRC,
            [f'Total Length of {d} Packets', f'{d} Packet Length Mean', n],
            lambda V, m, d=d, n=n: (_true(V[n]), _neq(V[f'Total Length of {d} Packets'], V[f'{d} Packet Length Mean'] * V[n])))
        add(f'CIC-A-avgseg=mean[{d}]', 'A', f'Avg {d} Segment Size = {d} Packet Length Mean', CIC_SRC,
            [f'Avg {d} Segment Size', f'{d} Packet Length Mean'],
            lambda V, m, d=d: (_true(V[F]), _neq(V[f'Avg {d} Segment Size'], V[f'{d} Packet Length Mean'], 1e-3, 1e-6)))
        add(f'CIC-A-rate[{d} Packets/s]', 'A', f'{d} Packets/s = packet count / Flow Duration (microseconds)', CIC_SRC,
            [f'{d} Packets/s', n, DUR],
            lambda V, m, d=d, n=n: (V[DUR] > 0, _neq(V[f'{d} Packets/s'], V[n] / np.where(V[DUR] > 0, V[DUR], 1) * 1e6, 1e-3, EQ_RTOL)))
        add(f'CIC-A-iattot=mean*(n-1)[{d}]', 'A', f'{d} IAT Total = {d} IAT Mean x (packet count - 1)', CIC_SRC,
            [f'{d} IAT Total', f'{d} IAT Mean', n],
            lambda V, m, d=d, n=n: (V[n] > 1, _neq(V[f'{d} IAT Total'], V[f'{d} IAT Mean'] * (V[n] - 1))))
    add('CIC-A-rate[Flow Packets/s]', 'A', 'Flow Packets/s = (fwd + bwd packets) / Flow Duration', CIC_SRC, ['Flow Packets/s', F, B, DUR],
        lambda V, m: (V[DUR] > 0, _neq(V['Flow Packets/s'], (V[F] + V[B]) / np.where(V[DUR] > 0, V[DUR], 1) * 1e6, 1e-3, EQ_RTOL)))
    add('CIC-A-rate[Flow Bytes/s]', 'A', 'Flow Bytes/s = (fwd + bwd bytes) / Flow Duration', CIC_SRC,
        ['Flow Bytes/s', 'Total Length of Fwd Packets', 'Total Length of Bwd Packets', DUR],
        lambda V, m: (V[DUR] > 0, _neq(V['Flow Bytes/s'], (V['Total Length of Fwd Packets'] + V['Total Length of Bwd Packets']) / np.where(V[DUR] > 0, V[DUR], 1) * 1e6, 1e-3, EQ_RTOL)))
    add('CIC-A-flowiatmean', 'A', 'Flow IAT Mean = Flow Duration / (total packets - 1)', CIC_SRC, ['Flow IAT Mean', DUR, F, B],
        lambda V, m: (V[F] + V[B] > 1, _neq(V['Flow IAT Mean'], V[DUR] / np.maximum(V[F] + V[B] - 1, 1))))
    add('CIC-A-var=std^2', 'A', 'Packet Length Variance = Packet Length Std squared', CIC_SRC, ['Packet Length Variance', 'Packet Length Std'],
        lambda V, m: (_true(V[F]), _neq(V['Packet Length Variance'], V['Packet Length Std'] ** 2, 1e-2, EQ_RTOL)))
    add('CIC-A-plmean=bytes/pkts', 'A', 'Packet Length Mean = total bytes / total packets', CIC_SRC,
        ['Packet Length Mean', 'Total Length of Fwd Packets', 'Total Length of Bwd Packets', F, B],
        lambda V, m: (V[F] + V[B] > 0, _neq(V['Packet Length Mean'], (V['Total Length of Fwd Packets'] + V['Total Length of Bwd Packets']) / np.maximum(V[F] + V[B], 1), 1e-2, EQ_RTOL)))
    add('CIC-A-avgpkt=bytes/pkts', 'A', 'Average Packet Size = total bytes / total packets', CIC_SRC,
        ['Average Packet Size', 'Total Length of Fwd Packets', 'Total Length of Bwd Packets', F, B],
        lambda V, m: (V[F] + V[B] > 0, _neq(V['Average Packet Size'], (V['Total Length of Fwd Packets'] + V['Total Length of Bwd Packets']) / np.maximum(V[F] + V[B], 1), 1e-2, EQ_RTOL)))
    add('CIC-A-maxpl=max', 'A', 'Max Packet Length = max(Fwd Packet Length Max, Bwd Packet Length Max)', CIC_SRC,
        ['Max Packet Length', 'Fwd Packet Length Max', 'Bwd Packet Length Max'],
        lambda V, m: (_true(V[F]), np.abs(V['Max Packet Length'] - np.maximum(V['Fwd Packet Length Max'], V['Bwd Packet Length Max'])) > INT_TOL))
    add('CIC-A-minpl=min', 'A', 'Min Packet Length = min(Fwd Packet Length Min, Bwd Packet Length Min) when backward packets exist', CIC_SRC,
        ['Min Packet Length', 'Fwd Packet Length Min', 'Bwd Packet Length Min', B],
        lambda V, m: (V[B] > 0, np.abs(V['Min Packet Length'] - np.minimum(V['Fwd Packet Length Min'], V['Bwd Packet Length Min'])) > INT_TOL))
    add('CIC-A-downup', 'A', 'Down/Up Ratio = floor(backward packets / forward packets)', CIC_SRC, ['Down/Up Ratio', F, B],
        lambda V, m: (V[F] > 0, np.abs(V['Down/Up Ratio'] - np.floor(V[B] / np.maximum(V[F], 1) + 1e-9)) > INT_TOL))
    # conditional zeros
    for f in CIC_BWD_ZERO:
        if f in features:
            add(f'CIC-C-nobwd[{f}]', 'C', f'No backward packets -> {f} = 0', CIC_SRC, [f, B],
                lambda V, m, f=f: (V[B] < 0.5, np.abs(V[f]) > 1e-9))
    for f in CIC_FWD_IAT:
        add(f'CIC-C-onefwd[{f}]', 'C', f'At most one forward packet -> {f} = 0 (no inter-arrival time exists)', CIC_SRC, [f, F],
            lambda V, m, f=f: (V[F] < 1.5, np.abs(V[f]) > 1e-9))
    for f in CIC_BWD_IAT:
        add(f'CIC-C-onebwd[{f}]', 'C', f'At most one backward packet -> {f} = 0', CIC_SRC, [f, B],
            lambda V, m, f=f: (V[B] < 1.5, np.abs(V[f]) > 1e-9))
    return C


# --------------------------------------------------------------------------------------------
# UNSW-NB15 (Argus / Bro-IDS features)
# --------------------------------------------------------------------------------------------
UNSW_SRC = 'NUSW-NB15_features.csv feature table (Moustafa & Slay, 2015)'
UNSW_INT = ['sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb',
            'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd',
            'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm',
            'ct_dst_sport_ltm', 'ct_dst_src_ltm']
UNSW_CT = ['ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm']
UNSW_TCP_ONLY = ['swin', 'dwin', 'stcpb', 'dtcpb', 'tcprtt', 'synack', 'ackdat']
UNSW_DST_ZERO = ['dbytes', 'dttl', 'dloss', 'Dload', 'dwin', 'dtcpb', 'dmeansz', 'Djit', 'Dintpkt']
UNSW_TCP_STATES = ['FIN', 'RST', 'ACC', 'CLO']                                   # Argus TCP connection states
UNSW_ICMP_STATES = ['ECO', 'ECR', 'MAS', 'PAR', 'TST', 'TXD', 'URH', 'URN']      # Argus ICMP message states
UNSW_NUM_NOT_CAT = None


def _code(m, feat, label):
    return m['codes'][feat].get(label, -999)


def _isin(arr, codes):
    return np.isin(np.round(arr).astype(np.int64), np.asarray(list(codes), dtype=np.int64))


def unsw_candidates(features, categorical=('proto', 'state', 'service')):
    C = []
    add = lambda *a: C.append(Constraint(*a))
    S, D, DUR = 'Spkts', 'Dpkts', 'dur'
    for f in features:
        if f in categorical: continue
        add(f'UNSW-D-nonneg[{f}]', 'D', f'{f} >= 0', UNSW_SRC, [f], lambda V, m, f=f: (_true(V[f]), V[f] < -1e-9))
    for f in UNSW_INT:
        add(f'UNSW-D-int[{f}]', 'D', f'{f} is integer-valued (type "integer"/"binary" in the feature table)', UNSW_SRC, [f],
            lambda V, m, f=f: (_true(V[f]), np.abs(V[f] - np.round(V[f])) > INT_TOL))
    for f in ('sttl', 'dttl'):
        add(f'UNSW-D-ttl[{f}]', 'D', f'{f} <= 255 (8-bit IP time-to-live)', UNSW_SRC, [f], lambda V, m, f=f: (_true(V[f]), V[f] > 255 + INT_TOL))
    for f in ('is_sm_ips_ports', 'is_ftp_login'):
        add(f'UNSW-D-binary[{f}]', 'D', f'{f} in {{0, 1}} (type "binary")', UNSW_SRC, [f],
            lambda V, m, f=f: (_true(V[f]), (np.abs(V[f]) > INT_TOL) & (np.abs(V[f] - 1) > INT_TOL)))
    for f in UNSW_CT:
        add(f'UNSW-D-ct[{f}]', 'D', f'{f} in [1, 100] (count over the last 100 connections, including the record itself)', UNSW_SRC, [f],
            lambda V, m, f=f: (_true(V[f]), (V[f] < 1 - INT_TOL) | (V[f] > 100 + INT_TOL)))
    add('UNSW-D-minpkt', 'D', 'A record contains at least one source packet', UNSW_SRC, [S], lambda V, m: (_true(V[S]), V[S] < 1 - INT_TOL))
    # ordering
    add('UNSW-O-sloss<=spkts', 'O', 'sloss <= Spkts', UNSW_SRC, ['sloss', S], lambda V, m: (_true(V[S]), V['sloss'] > V[S] + INT_TOL))
    add('UNSW-O-dloss<=dpkts', 'O', 'dloss <= Dpkts', UNSW_SRC, ['dloss', D], lambda V, m: (_true(V[D]), V['dloss'] > V[D] + INT_TOL))
    add('UNSW-O-sbytes>=spkts', 'O', 'sbytes >= Spkts (every packet carries at least one byte)', UNSW_SRC, ['sbytes', S],
        lambda V, m: (_true(V[S]), V['sbytes'] < V[S] - INT_TOL))
    add('UNSW-O-dbytes>=dpkts', 'O', 'dbytes >= Dpkts', UNSW_SRC, ['dbytes', D], lambda V, m: (_true(V[D]), V['dbytes'] < V[D] - INT_TOL))
    add('UNSW-O-srcdport<=src', 'O', 'ct_src_dport_ltm <= ct_src_ltm (same source AND port is a subset of same source)', UNSW_SRC,
        ['ct_src_dport_ltm', 'ct_src_ltm'], lambda V, m: (_true(V[S]), V['ct_src_dport_ltm'] > V['ct_src_ltm'] + INT_TOL))
    add('UNSW-O-dstsport<=dst', 'O', 'ct_dst_sport_ltm <= ct_dst_ltm', UNSW_SRC, ['ct_dst_sport_ltm', 'ct_dst_ltm'],
        lambda V, m: (_true(V[S]), V['ct_dst_sport_ltm'] > V['ct_dst_ltm'] + INT_TOL))
    add('UNSW-O-dstsrc<=dst', 'O', 'ct_dst_src_ltm <= ct_dst_ltm', UNSW_SRC, ['ct_dst_src_ltm', 'ct_dst_ltm'],
        lambda V, m: (_true(V[S]), V['ct_dst_src_ltm'] > V['ct_dst_ltm'] + INT_TOL))
    add('UNSW-O-dstsrc<=src', 'O', 'ct_dst_src_ltm <= ct_src_ltm', UNSW_SRC, ['ct_dst_src_ltm', 'ct_src_ltm'],
        lambda V, m: (_true(V[S]), V['ct_dst_src_ltm'] > V['ct_src_ltm'] + INT_TOL))
    add('UNSW-O-tcprtt<=dur', 'O', 'tcprtt <= dur for TCP records (handshake lies inside the record)', UNSW_SRC, ['tcprtt', DUR, 'proto'],
        lambda V, m: (_isin(V['proto'], [_code(m, 'proto', 'tcp')]), V['tcprtt'] > V[DUR] + 1e-4))
    for d, n, ip in (('S', S, 'Sintpkt'), ('D', D, 'Dintpkt')):
        add(f'UNSW-O-intpkt<=dur[{d}]', 'O', f'{ip} x (packets - 1) <= dur (ms vs s)', UNSW_SRC, [ip, n, DUR],
            lambda V, m, n=n, ip=ip: (V[n] > 1, V[ip] * (V[n] - 1) > V[DUR] * 1000 * 1.001 + 1e-3))
    # arithmetic
    add('UNSW-A-tcprtt', 'A', 'tcprtt = synack + ackdat (stated in the feature table)', UNSW_SRC, ['tcprtt', 'synack', 'ackdat'],
        lambda V, m: (_true(V['tcprtt']), np.abs(V['tcprtt'] - V['synack'] - V['ackdat']) > 1e-5))
    add('UNSW-A-smeansz', 'A', '|smeansz - sbytes/Spkts| <= 1 (integer mean packet size)', UNSW_SRC, ['smeansz', 'sbytes', S],
        lambda V, m: (V[S] > 0, np.abs(V['smeansz'] - V['sbytes'] / np.maximum(V[S], 1)) > 1 + 1e-6))
    add('UNSW-A-dmeansz', 'A', '|dmeansz - dbytes/Dpkts| <= 1', UNSW_SRC, ['dmeansz', 'dbytes', D],
        lambda V, m: (V[D] > 0, np.abs(V['dmeansz'] - V['dbytes'] / np.maximum(V[D], 1)) > 1 + 1e-6))
    add('UNSW-A-sload', 'A', 'Sload = 8 x sbytes / dur ("source bits per second")', UNSW_SRC, ['Sload', 'sbytes', DUR],
        lambda V, m: (V[DUR] > 0, _neq(V['Sload'], 8 * V['sbytes'] / np.where(V[DUR] > 0, V[DUR], 1), 1.0, 1e-2)))
    add('UNSW-A-dload', 'A', 'Dload = 8 x dbytes / dur', UNSW_SRC, ['Dload', 'dbytes', DUR],
        lambda V, m: (V[DUR] > 0, _neq(V['Dload'], 8 * V['dbytes'] / np.where(V[DUR] > 0, V[DUR], 1), 1.0, 1e-2)))
    # conditional zeros
    for f in UNSW_DST_ZERO:
        add(f'UNSW-C-nodst[{f}]', 'C', f'No destination packets -> {f} = 0', UNSW_SRC, [f, D], lambda V, m, f=f: (V[D] < 0.5, np.abs(V[f]) > 1e-9))
    add('UNSW-C-onedst[Dintpkt]', 'C', 'At most one destination packet -> Dintpkt = 0', UNSW_SRC, ['Dintpkt', D], lambda V, m: (V[D] < 1.5, np.abs(V['Dintpkt']) > 1e-9))
    add('UNSW-C-onesrc[Sintpkt]', 'C', 'At most one source packet -> Sintpkt = 0', UNSW_SRC, ['Sintpkt', S], lambda V, m: (V[S] < 1.5, np.abs(V['Sintpkt']) > 1e-9))
    add('UNSW-C-twosrc[Sjit]', 'C', 'At most two source packets -> Sjit = 0 (jitter needs two inter-arrival gaps)', UNSW_SRC, ['Sjit', S], lambda V, m: (V[S] < 2.5, np.abs(V['Sjit']) > 1e-9))
    add('UNSW-C-twodst[Djit]', 'C', 'At most two destination packets -> Djit = 0', UNSW_SRC, ['Djit', D], lambda V, m: (V[D] < 2.5, np.abs(V['Djit']) > 1e-9))
    add('UNSW-C-onepkt[dur]', 'C', 'A single-packet record has dur = 0', UNSW_SRC, [DUR, S, D], lambda V, m: (V[S] + V[D] < 1.5, np.abs(V[DUR]) > 1e-9))
    for f in ('trans_depth', 'res_bdy_len', 'ct_flw_http_mthd'):
        add(f'UNSW-C-nonhttp[{f}]', 'C', f'service != http -> {f} = 0 (HTTP-only field)', UNSW_SRC, [f, 'service'],
            lambda V, m, f=f: (~_isin(V['service'], [_code(m, 'service', 'http')]), np.abs(V[f]) > 1e-9))
    for f in ('is_ftp_login', 'ct_ftp_cmd'):
        add(f'UNSW-C-nonftp[{f}]', 'C', f'service != ftp -> {f} = 0 (FTP-session field)', UNSW_SRC, [f, 'service'],
            lambda V, m, f=f: (~_isin(V['service'], [_code(m, 'service', 'ftp')]), np.abs(V[f]) > 1e-9))
    # protocol-dependent fields
    for f in UNSW_TCP_ONLY:
        add(f'UNSW-P-nontcp[{f}]', 'P', f'proto != tcp -> {f} = 0 (TCP-only field)', UNSW_SRC, [f, 'proto'],
            lambda V, m, f=f: (~_isin(V['proto'], [_code(m, 'proto', 'tcp')]), np.abs(V[f]) > 1e-9))
    add('UNSW-P-tcpstate', 'P', 'state in {FIN, RST, ACC, CLO} -> proto = tcp (Argus TCP states)', UNSW_SRC + '; Argus ra(1) state field', ['state', 'proto'],
        lambda V, m: (_isin(V['state'], [_code(m, 'state', s) for s in UNSW_TCP_STATES]), ~_isin(V['proto'], [_code(m, 'proto', 'tcp')])))
    add('UNSW-P-icmpstate', 'P', 'state in {ECO, ECR, MAS, PAR, TST, TXD, URH, URN} -> proto = icmp (Argus ICMP states)', UNSW_SRC + '; Argus ra(1) state field', ['state', 'proto'],
        lambda V, m: (_isin(V['state'], [_code(m, 'state', s) for s in UNSW_ICMP_STATES]), ~_isin(V['proto'], [_code(m, 'proto', 'icmp')])))
    return C


def candidates(ds, features):
    return cic_candidates(features) if ds == 'cic' else unsw_candidates(features)


def evaluate(constraints, X, features, meta):
    """Return boolean matrix viol[n, n_constraints] and applicable[n, n_constraints] for rows of X."""
    V = {f: X[..., j] for j, f in enumerate(features)}
    viol = np.zeros(X.shape[:-1] + (len(constraints),), dtype=bool)
    app = np.zeros_like(viol)
    for k, c in enumerate(constraints):
        a, v = c.check(V, meta)
        app[..., k], viol[..., k] = a, v
    return viol, app

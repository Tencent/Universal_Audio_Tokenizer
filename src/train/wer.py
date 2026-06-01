import re


def text2tokens(text, do_tn=False):
    if do_tn:
        import cn2an
        text = cn2an.transform(text, "an2cn")
    PUNCTUATIONS = (
        "，。？！,\.?!＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·｡\"':"
        + "()\[\]{}/;`|=+"
    )
    if text == "":
        return []
    tokens = []

    text = re.sub("<unk>", "", text)
    text = re.sub(r"[%s]+" % PUNCTUATIONS, " ", text)

    pattern = re.compile(r'([\u4e00-\u9fff])')
    parts = pattern.split(text.strip().lower())
    parts = [p for p in parts if len(p.strip()) > 0]
    for part in parts:
        if pattern.fullmatch(part) is not None:
            tokens.append(part)
        else:
            for word in part.strip().split():
                tokens.append(word)
    return tokens


COST_SUB = 3
COST_DEL = 3
COST_INS = 3

ALIGN_CRT = 0
ALIGN_SUB = 1
ALIGN_DEL = 2
ALIGN_INS = 3
ALIGN_END = 4


def compute_one_wer_info(ref, hyp):
    """Impl minimum edit distance and backtrace.
    Args:
        ref, hyp: List[str]
    Returns:
        WerInfo
    """
    ref_len = len(ref)
    hyp_len = len(hyp)

    class _DpPoint:
        def __init__(self, cost, align):
            self.cost = cost
            self.align = align

    dp = []
    for i in range(0, ref_len + 1):
        dp.append([])
        for j in range(0, hyp_len + 1):
            dp[-1].append(_DpPoint(i * j, ALIGN_CRT))

    # Initialize
    for i in range(1, hyp_len + 1):
        dp[0][i].cost = dp[0][i - 1].cost + COST_INS;
        dp[0][i].align = ALIGN_INS
    for i in range(1, ref_len + 1):
        dp[i][0].cost = dp[i - 1][0].cost + COST_DEL
        dp[i][0].align = ALIGN_DEL

    # DP
    for i in range(1, ref_len + 1):
        for j in range(1, hyp_len + 1):
            min_cost = 0
            min_align = ALIGN_CRT
            if hyp[j - 1] == ref[i - 1]:
                min_cost = dp[i - 1][j - 1].cost
                min_align = ALIGN_CRT
            else:
                min_cost = dp[i - 1][j - 1].cost + COST_SUB
                min_align = ALIGN_SUB

            del_cost = dp[i - 1][j].cost + COST_DEL
            if del_cost < min_cost:
                min_cost = del_cost
                min_align = ALIGN_DEL

            ins_cost = dp[i][j - 1].cost + COST_INS
            if ins_cost < min_cost:
                min_cost = ins_cost
                min_align = ALIGN_INS

            dp[i][j].cost = min_cost
            dp[i][j].align = min_align

    # Backtrace
    crt = sub = ins = det = 0
    i = ref_len
    j = hyp_len
    align = []
    while i > 0 or j > 0:
        if dp[i][j].align == ALIGN_CRT:
            align.append((i, j, ALIGN_CRT))
            i -= 1
            j -= 1
            crt += 1
        elif dp[i][j].align == ALIGN_SUB:
            align.append((i, j, ALIGN_SUB))
            i -= 1
            j -= 1
            sub += 1
        elif dp[i][j].align == ALIGN_DEL:
            align.append((i, j, ALIGN_DEL))
            i -= 1
            det += 1
        elif dp[i][j].align == ALIGN_INS:
            align.append((i, j, ALIGN_INS))
            j -= 1
            ins += 1

    err = sub + det + ins
    align.reverse()
    wer_info = WerInfo(ref_len, err, crt, sub, det, ins, align)
    return wer_info


class WerInfo:
    def __init__(self, ref, err, crt, sub, dele, ins, ali):
        self.r = ref
        self.e = err
        self.c = crt
        self.s = sub
        self.d = dele
        self.i = ins
        self.ali = ali
        r = max(self.r, 1)
        self.wer = 100.0 * (self.s + self.d + self.i) / r

    def __repr__(self):
        s = f"wer {self.wer:.2f}% ref {self.r:2d} sub {self.s:2d} del {self.d:2d} ins {self.i:2d}"
        return s


class WerStats:
    def __init__(self):
        self.infos = []

    def add(self, wer_info):
        self.infos.append(wer_info)

    def print(self):
        r = self.total_ref()
        if r <= 0:
            print(f"REF len is {r}, check")
            r = 1
        s = self.total_sub()
        d = self.total_del()
        i = self.total_ins()
        se = 100.0 * s / r
        de = 100.0 * d / r
        ie = 100.0 * i / r
        wer = 100.0 * (s + d + i) / r
        sen = max(len(self.infos), 1)
        errsen = sum(info.e > 0 for info in self.infos)
        ser = 100.0 * errsen / sen
        print("-"*80)
        print(f"ref{r:6d}  sub{s:6d}  del{d:6d}  ins{i:6d} ")
        print(f"WER{wer:6.2f}% sub{se:6.2f}% del{de:6.2f}% ins{ie:6.2f}%")
        print(f"SER{ser:6.2f}% = {errsen} / {sen}")
        print("-"*80)

    def total_ref(self):
        return sum(info.r for info in self.infos)

    def total_sub(self):
        return sum(info.s for info in self.infos)

    def total_del(self):
        return sum(info.d for info in self.infos)

    def total_ins(self):
        return sum(info.i for info in self.infos)


class EnDigStats:
    def __init__(self):
        self.n_en_word = 0
        self.n_en_correct = 0
        self.n_dig_word = 0
        self.n_dig_correct = 0

    def add(self, n_en_word, n_en_correct, n_dig_word, n_dig_correct):
        self.n_en_word += n_en_word
        self.n_en_correct += n_en_correct
        self.n_dig_word += n_dig_word
        self.n_dig_correct += n_dig_correct

    def print(self):
        print(f"English #word={self.n_en_word}, #correct={self.n_en_correct}\n"
              f"Digit #word={self.n_dig_word}, #correct={self.n_dig_correct}")
        print("-"*80)


def count_english_ditgit(ref, hyp, wer_info):
    patt_en = "[a-zA-Z\.\-\']+"
    patt_dig = "[0-9]+"
    patt_cjk = re.compile(r'([\u4e00-\u9fff])')
    n_en_word = 0
    n_en_correct = 0
    n_dig_word = 0
    n_dig_correct = 0
    ali = wer_info.ali
    for i, token in enumerate(ref):
        if re.match(patt_en, token):
            n_en_word += 1
            for y in ali:
                if y[0] == i+1 and y[2] == ALIGN_CRT:
                    j = y[1] - 1
                    n_en_correct += 1
                    break
        if re.match(patt_dig, token):
            n_dig_word += 1
            for y in ali:
                if y[0] == i+1 and y[2] == ALIGN_CRT:
                    j = y[1] - 1
                    n_dig_correct += 1
                    break
        if not re.match(patt_cjk, token) and not re.match(patt_en, token) \
           and not re.match(patt_dig, token):
            print("[WiredChar]:", [token])
    return n_en_word, n_en_correct, n_dig_word, n_dig_correct

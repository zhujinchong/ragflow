import re


def rm_stop_words(txt):
    """删除停用词"""
    patts = [
        (
            r"[ \r\n\t,\./;'\[\]\\`~!@#$%\^&\*\(\)=\+_<>\?:\"\{\}\|，。；‘’【】、！￥……（）——《》？：“”-]+",
            " "
        ),
        (
            r"是*(什么样的|哪家|一下|那家|啥样|咋样了|什么时候|何时|何地|何人|是否|是不是|多少|哪里|怎么|哪儿|怎么样|如何|哪些|是啥|啥是|啊|吗|呢|吧|咋|什么|有没有|呀)是*",
            ""
        ),
        (
            r"(^| )(what|who|how|which|where|why)('re|'s)? ", " "
        ),
        (
            r"(^| )('s|'re|is|are|were|was|do|does|did|don't|doesn't|didn't|has|have|be|there|you|me|your|my|mine|just|please|may|i|should|would|wouldn't|will|won't|done|go|for|with|so|the|a|an|by|i'm|it's|he's|she's|they|they're|you're|as|by|on|in|at|up|out|down) ",
            " "
        )
    ]
    for r, p in patts:
        txt = re.sub(r, p, txt, flags=re.IGNORECASE)
    return txt


def escape_char(line):
    """转义字符"""
    return re.sub(r"([:\{\}/\[\]\-\*\"\(\)\|~\^])", r"\\\1", line).strip()

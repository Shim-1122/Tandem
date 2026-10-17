"""Languages: which one you are speaking, how to find "Jarvis" in it, and its words for the commands.

How the language is chosen (see `wake_language`):
    1. The words around "Jarvis" in the wake phrase, when they belong to one language: "Wake up Jarvis" is English,
       "Despierta Jarvis" Spanish, "Réveille-toi Jarvis" French, "ジャービス、起きて" Japanese, "जार्विस, उठो" Hindi.
    2. Otherwise ("Hey Jarvis" is said in many languages) the language Whisper detects in the whole sentence.
    3. Otherwise the language of the last conversation.

Coverage:
    understood  every language Whisper knows (about 99), through the language model
    built in    12 languages with hand-written command words and replies, working without any model:
                English, Spanish, French, German, Italian, Portuguese, Hindi, Chinese, Japanese, Korean, Russian,
                Arabic
    spoken      every language with a Piper voice (about 50; `voice_for`); others are shown as text only
    stop        "stop" in about 35 languages, always handled without the model (safety)
"""
from __future__ import annotations

import re

NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German", "it": "Italian", "pt": "Portuguese",
    "hi": "Hindi", "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ru": "Russian", "ar": "Arabic",
    "nl": "Dutch", "pl": "Polish", "tr": "Turkish", "sv": "Swedish", "uk": "Ukrainian", "vi": "Vietnamese",
    "id": "Indonesian", "el": "Greek", "he": "Hebrew", "fa": "Persian", "cs": "Czech", "ro": "Romanian",
    "hu": "Hungarian", "th": "Thai", "bn": "Bengali", "te": "Telugu", "ta": "Tamil", "ml": "Malayalam",
    "mr": "Marathi", "ur": "Urdu", "ne": "Nepali", "fi": "Finnish", "da": "Danish", "no": "Norwegian",
    "ca": "Catalan", "bg": "Bulgarian", "sk": "Slovak", "sl": "Slovenian", "sr": "Serbian", "hr": "Croatian",
    "lt": "Lithuanian", "lv": "Latvian", "et": "Estonian", "ka": "Georgian", "kk": "Kazakh", "sw": "Swahili",
    "is": "Icelandic", "cy": "Welsh", "eu": "Basque", "hy": "Armenian", "sq": "Albanian", "lb": "Luxembourgish",
    "ms": "Malay", "tl": "Tagalog", "gu": "Gujarati", "kn": "Kannada", "pa": "Punjabi", "af": "Afrikaans",
}
BUILT_IN = ("en", "es", "fr", "de", "it", "pt", "hi", "zh", "ja", "ko", "ru", "ar")

# "Jarvis" as Whisper writes it in different scripts, including the near misses it makes when the name is said
# with another language's sounds (heard in testing: "Harvis", "Жарvis", "じゃあビス", "जारवेस", "جار فيس").
JARVIS = (r"(?<![a-zа-яё])(?:дж|ж|dzh|ch|[jyhgžʒ])[aаäæeо]+[rр][vвbwфβ][iиіyeе][sсzшš](?![a-zа-яё])|"
          r"jarvis(?![a-z])|"                    # glued to a misheard word before it: "o dejarvis" for "oye Jarvis"
          r"ジャービス|ジャーヴィス|ジャビス|ジャ[ーｰ]?ヴィス|チャービス|じゃ[あー]?び[すず]|じゃあビス|"
          r"贾维斯|賈維斯|杰维斯|자비스|재비스|"
          r"जार्विस|जारविस|जारवेस|जार्वेस|जार्विज़|"
          r"جار ?فيس|جارفس|جارویس|ג'רוויס|ג׳רוויס|τζάρβις|จาร์วิส|ஜார்விஸ்|జార్విస్|ജാർവിസ്|জার্ভিস|જાર્વિસ|ಜಾರ್ವಿಸ್")
JARVIS_RE = re.compile(JARVIS, re.I)

# Wake words that belong to one language, said just before or after "Jarvis". Shared ones ("hey") are left out.
WAKE_WORDS = {
    "en": ["wake up", "hello", "good morning", "yo", "okay", "ok", "hi"],
    "es": ["despierta", "oye", "hola", "buenos días", "buenas"],
    "fr": ["réveille-toi", "réveille toi", "réveille", "salut", "bonjour", "dis donc", "coucou"],
    "de": ["wach auf", "wach", "aufwachen", "hallo", "guten morgen", "servus", "moin"],
    "it": ["svegliati", "sveglia", "ciao", "ehi", "buongiorno"],
    "pt": ["acorda", "olá", "ola", "oi", "bom dia"],
    "hi": ["उठो", "उठ जाओ", "जागो", "अरे", "नमस्ते", "सुनो", "utho", "jaago", "arre", "suno"],
    "zh": ["醒醒", "醒来", "起床", "你好", "喂", "嘿"],
    "ja": ["起きて", "おきて", "ねえ", "ねぇ", "おい", "やあ", "こんにちは", "おはよう"],
    "ko": ["일어나", "헤이", "야", "안녕", "여보세요"],
    "ru": ["проснись", "просыпайся", "привет", "эй", "слушай"],
    "ar": ["استيقظ", "اصح", "مرحبا", "يا"],
    "nl": ["word wakker", "wakker", "hoi", "hallo daar"],
    "pl": ["obudź się", "obudz sie", "cześć", "hej"],
    "tr": ["uyan", "merhaba", "selam"],
    "sv": ["vakna", "hej hej"],
    "uk": ["прокинься", "привіт", "гей"],
    "vi": ["dậy đi", "dậy", "xin chào", "chào"],
    "id": ["bangun", "halo"],
    "el": ["ξύπνα", "γεια"],
    "he": ["תתעורר", "שלום"],
    "fa": ["بیدار شو", "سلام"],
    "ta": ["எழுந்திரு", "வணக்கம்"],
    "te": ["లే", "నమస్కారం"],
    "bn": ["ওঠো", "নমস্কার"],
}

# Emergency stop, checked on every utterance before anything else (no model involved)
STOP = {
    "en": r"\b(stop|freeze|halt|hold still|don'?t move|emergency)\b",
    "es": r"\b(alto|detente|deténte|quieto|quieta|para ya|párate|basta)\b",
    "fr": r"\b(arrête|arrêtes|arrêtez|stoppe|halte|bouge plus|ne bouge pas)\b",
    "de": r"\b(stopp|halt|anhalten|hör auf|nicht bewegen|stehen bleiben)\b",
    "it": r"\b(fermo|ferma|fermati|fermatevi|basta|alt)\b",
    "pt": r"\b(pare|para já|parado|parada|chega|basta)\b",
    "hi": r"(रुको|रुक जाओ|ठहरो|बस करो|रोको|ruko|ruk jao|bas karo)",
    "zh": r"(停|停下|别动|不要动|住手)",
    "ja": r"(止まれ|とまれ|止まって|とまって|やめて|ストップ|動くな)",
    "ko": r"(멈춰|그만|정지|스톱|움직이지 마)",
    "ru": r"(стоп|стой|замри|хватит|остановись|не двигайся)",
    "ar": r"(توقف|قف|كفى|لا تتحرك)",
    "nl": r"\b(stoppen|halt|niet bewegen|sta stil)\b",
    "pl": r"\b(stój|stop|przestań|nie ruszaj się)\b",
    "tr": r"\b(dur|durun|kıpırdama)\b",
    "sv": r"\b(stanna|sluta|stopp)\b",
    "da": r"\b(stands|stop|hold op)\b",
    "no": r"\b(stans|stopp|slutt)\b",
    "fi": r"\b(seis|pysähdy|lopeta)\b",
    "uk": r"(стоп|стій|зупинись|досить)",
    "cs": r"\b(stůj|zastav|přestaň)\b",
    "sk": r"\b(stoj|zastav|prestaň)\b",
    "ro": r"\b(oprește|stai|gata)\b",
    "hu": r"\b(állj|elég|ne mozdulj)\b",
    "el": r"(σταμάτα|στοπ|ακίνητος)",
    "he": r"(עצור|תעצור|די)",
    "fa": r"(بایست|توقف کن|بسه)",
    "tr2": r"\b(yeter)\b",
    "vi": r"\b(dừng lại|đứng yên|dừng)\b",
    "id": r"\b(berhenti|stop|diam)\b",
    "th": r"(หยุด|พอแล้ว)",
    "bn": r"(থামো|থামুন)",
    "ta": r"(நிறுத்து|நில்)",
    "te": r"(ఆపు|ఆగు)",
    "ml": r"(നിർത്ത്|നിൽക്ക്)",
    "mr": r"(थांब|थांबा)",
    "ur": r"(رکو|بس کرو)",
}
STOP_RE = re.compile("|".join(f"(?:{p})" for p in STOP.values()), re.I)

# Words for the commands, translated into the English the rule parser understands. Longest phrases first.
COMMANDS = {
    "es": [("deja de imitarme", "stop mirroring me"), ("deja de copiarme", "stop copying me"),
           ("piedra, papel o tijera", "rock paper scissors"), ("piedra papel o tijera", "rock paper scissors"),
           ("piedra, papel, tijera", "rock paper scissors"), ("piedra papel tijera", "rock paper scissors"),
           ("imítame", "mirror me"), ("imitame", "mirror me"), ("sígueme", "shadow me"), ("cópiame", "copy me"),
           ("copiame", "copy me"), ("saluda", "wave"), ("pulgar arriba", "thumbs up"), ("asiente", "nod"),
           ("di que sí", "nod"), ("niega con la cabeza", "shake your head"), ("señala", "point"), ("apunta", "point"),
           ("mírame", "look at me"), ("mira", "look"), ("a mí", "at me"), ("dámelo", "give it to me"),
           ("suelta", "let go"), ("agarra", "grip"), ("aprieta", "grip"), ("suavemente", "gently"),
           ("con cuidado", "gently"), ("celebra", "celebrate"), ("encógete de hombros", "shrug"),
           ("relájate", "relax"), ("descansa", "rest"), ("izquierda", "left"), ("derecha", "right"),
           ("arriba", "up"), ("abajo", "down"), ("mano", "hand"), ("hola", "hello"),
           ("qué tengo en la mano", "what am i holding"), ("qué ves", "what do you see")],
    "fr": [("arrête de m'imiter", "stop mirroring me"), ("arrête de me copier", "stop copying me"),
           ("pierre, feuille, ciseaux", "rock paper scissors"), ("pierre feuille ciseaux", "rock paper scissors"),
           ("chifoumi", "rock paper scissors"), ("imite-moi", "mirror me"), ("imite moi", "mirror me"),
           ("suis-moi", "shadow me"), ("copie-moi", "copy me"), ("copie moi", "copy me"), ("salue", "wave"),
           ("fais coucou", "wave"), ("pouce levé", "thumbs up"), ("hoche la tête", "nod"),
           ("secoue la tête", "shake your head"), ("montre", "point"), ("pointe", "point"),
           ("regarde-moi", "look at me"), ("regarde", "look"), ("donne-le-moi", "give it to me"),
           ("lâche", "let go"), ("attrape", "grip"), ("serre", "grip"), ("doucement", "gently"),
           ("célèbre", "celebrate"), ("hausse les épaules", "shrug"), ("détends-toi", "relax"),
           ("repose-toi", "rest"), ("gauche", "left"), ("droite", "right"), ("en haut", "up"), ("en bas", "down"),
           ("main", "hand"), ("bonjour", "hello"), ("qu'est-ce que je tiens", "what am i holding"),
           ("que vois-tu", "what do you see")],
    "de": [("hör auf mich nachzumachen", "stop mirroring me"), ("nicht mehr nachmachen", "stop mirroring me"),
           ("schere, stein, papier", "rock paper scissors"), ("schere stein papier", "rock paper scissors"),
           ("mach mich nach", "mirror me"), ("mach mir nach", "mirror me"), ("spiegel mich", "mirror me"),
           ("kopier mich", "copy me"), ("folge mir", "shadow me"), ("wink", "wave"), ("daumen hoch", "thumbs up"),
           ("nick", "nod"), ("schüttel den kopf", "shake your head"), ("zeig", "point"), ("schau mich an", "look at me"),
           ("schau", "look"), ("gib es mir", "give it to me"), ("lass los", "let go"), ("greif", "grip"),
           ("drück", "grip"), ("sanft", "gently"), ("vorsichtig", "gently"), ("feier", "celebrate"),
           ("zuck mit den schultern", "shrug"), ("entspann dich", "relax"), ("ruh dich aus", "rest"),
           ("links", "left"), ("rechts", "right"), ("oben", "up"), ("unten", "down"), ("hand", "hand"),
           ("hallo", "hello"), ("was halte ich", "what am i holding"), ("was siehst du", "what do you see")],
    "it": [("smetti di imitarmi", "stop mirroring me"), ("smetti di copiarmi", "stop copying me"),
           ("sasso, carta, forbice", "rock paper scissors"), ("sasso carta forbice", "rock paper scissors"),
           ("carta forbice sasso", "rock paper scissors"), ("morra cinese", "rock paper scissors"),
           ("imitami", "mirror me"), ("seguimi", "shadow me"), ("copiami", "copy me"), ("saluta", "wave"),
           ("pollice in su", "thumbs up"), ("annuisci", "nod"), ("scuoti la testa", "shake your head"),
           ("indica", "point"), ("guardami", "look at me"), ("guarda", "look"), ("dammelo", "give it to me"),
           ("lascia", "let go"), ("afferra", "grip"), ("stringi", "grip"), ("delicatamente", "gently"),
           ("piano", "gently"), ("festeggia", "celebrate"), ("alza le spalle", "shrug"), ("rilassati", "relax"),
           ("riposa", "rest"), ("sinistra", "left"), ("destra", "right"), ("su", "up"), ("giù", "down"),
           ("mano", "hand"), ("ciao", "hello"), ("cosa tengo in mano", "what am i holding"),
           ("cosa vedi", "what do you see")],
    "pt": [("para de me imitar", "stop mirroring me"), ("pare de me imitar", "stop mirroring me"),
           ("para de me copiar", "stop copying me"), ("pedra, papel, tesoura", "rock paper scissors"),
           ("pedra papel tesoura", "rock paper scissors"), ("pedra, papel e tesoura", "rock paper scissors"),
           ("jokenpô", "rock paper scissors"), ("jokenpo", "rock paper scissors"), ("me imita", "mirror me"),
           ("imite-me", "mirror me"), ("me segue", "shadow me"), ("me copia", "copy me"), ("acene", "wave"),
           ("dá tchau", "wave"), ("joinha", "thumbs up"), ("polegar para cima", "thumbs up"),
           ("balance a cabeça", "nod"), ("acene com a cabeça", "nod"), ("aponte", "point"),
           ("olhe para mim", "look at me"), ("olha pra mim", "look at me"), ("olhe", "look"),
           ("me dá", "give it to me"), ("solte", "let go"), ("solta", "let go"), ("segure", "grip"),
           ("aperte", "grip"), ("devagar", "gently"), ("comemore", "celebrate"), ("dê de ombros", "shrug"),
           ("relaxe", "relax"), ("descanse", "rest"), ("esquerda", "left"), ("direita", "right"),
           ("para cima", "up"), ("para baixo", "down"), ("mão", "hand"), ("olá", "hello"),
           ("o que estou segurando", "what am i holding"), ("o que você vê", "what do you see")],
    "hi": [("नकल करना बंद करो", "stop mirroring me"), ("मेरी नकल करो", "mirror me"), ("नकल करो", "mirror me"),
           ("मेरे पीछे चलो", "shadow me"), ("पत्थर कागज़ कैंची", "rock paper scissors"),
           ("पत्थर कागज कैंची", "rock paper scissors"), ("हाथ हिलाओ", "wave"), ("अंगूठा दिखाओ", "thumbs up"),
           ("सिर हिलाओ", "nod"), ("इशारा करो", "point"), ("मेरी तरफ देखो", "look at me"), ("देखो", "look"),
           ("छोड़ दो", "let go"), ("छोड़ो", "let go"), ("पकड़ो", "grip"), ("धीरे से", "gently"),
           ("जश्न मनाओ", "celebrate"), ("आराम करो", "relax"), ("बाएं", "left"), ("बाएँ", "left"),
           ("दाएं", "right"), ("दाएँ", "right"), ("ऊपर", "up"), ("नीचे", "down"), ("हाथ", "hand"),
           ("नमस्ते", "hello"), ("मैंने क्या पकड़ा है", "what am i holding"), ("तुम क्या देखते हो", "what do you see"),
           ("nakal karo", "mirror me"), ("haath hilao", "wave")],
    "zh": [("别模仿我了", "stop mirroring me"), ("不要模仿我了", "stop mirroring me"), ("停止模仿", "stop mirroring me"),
           ("石头剪刀布", "rock paper scissors"), ("剪刀石头布", "rock paper scissors"), ("模仿我", "mirror me"),
           ("学我", "mirror me"), ("跟着我", "shadow me"), ("复制我", "copy me"), ("挥手", "wave"),
           ("点赞", "thumbs up"), ("竖大拇指", "thumbs up"), ("点头", "nod"), ("摇头", "shake your head"),
           ("指向", "point"), ("指一下", "point"), ("看着我", "look at me"), ("看", "look"), ("给我", "give it to me"),
           ("松手", "let go"), ("放开", "let go"), ("抓住", "grip"), ("握紧", "grip"), ("轻轻", "gently"),
           ("庆祝", "celebrate"), ("耸肩", "shrug"), ("放松", "relax"), ("休息", "rest"), ("左", " left "),
           ("右", " right "), ("上", " up "), ("下", " down "), ("手", " hand "), ("你好", "hello"),
           ("我拿着什么", "what am i holding"), ("你看到什么", "what do you see")],
    "ja": [("真似するのをやめて", "stop mirroring me"), ("まねするのをやめて", "stop mirroring me"),
           ("じゃんけん", "rock paper scissors"), ("ジャンケン", "rock paper scissors"), ("真似して", "mirror me"),
           ("まねして", "mirror me"), ("ついてきて", "shadow me"), ("コピーして", "copy me"), ("手を振って", "wave"),
           ("いいね", "thumbs up"), ("うなずいて", "nod"), ("首を振って", "shake your head"), ("指さして", "point"),
           ("指差して", "point"), ("こっちを見て", "look at me"), ("私を見て", "look at me"), ("見て", "look"),
           ("ちょうだい", "give it to me"), ("離して", "let go"), ("放して", "let go"), ("つかんで", "grip"),
           ("握って", "grip"), ("優しく", "gently"), ("そっと", "gently"), ("お祝いして", "celebrate"),
           ("肩をすくめて", "shrug"), ("リラックスして", "relax"), ("休んで", "rest"), ("左", " left "),
           ("右", " right "), ("上", " up "), ("下", " down "), ("手", " hand "), ("こんにちは", "hello"),
           ("何を持ってる", "what am i holding"), ("何が見える", "what do you see")],
    "ko": [("따라하지 마", "stop mirroring me"), ("그만 따라해", "stop mirroring me"), ("가위바위보", "rock paper scissors"),
           ("따라해", "mirror me"), ("흉내 내", "mirror me"), ("복사해", "copy me"), ("손 흔들어", "wave"),
           ("엄지 척", "thumbs up"), ("끄덕여", "nod"), ("고개를 저어", "shake your head"), ("가리켜", "point"),
           ("나를 봐", "look at me"), ("날 봐", "look at me"), ("봐", "look"), ("줘", "give it to me"),
           ("놔", "let go"), ("놓아", "let go"), ("잡아", "grip"), ("부드럽게", "gently"), ("살살", "gently"),
           ("축하해", "celebrate"), ("어깨를 으쓱", "shrug"), ("쉬어", "relax"), ("왼쪽", " left "),
           ("오른쪽", " right "), ("위", " up "), ("아래", " down "), ("손", " hand "), ("안녕", "hello"),
           ("내가 뭘 들고 있어", "what am i holding"), ("뭐가 보여", "what do you see")],
    "ru": [("перестань повторять", "stop mirroring me"), ("хватит повторять", "stop mirroring me"),
           ("камень, ножницы, бумага", "rock paper scissors"), ("камень ножницы бумага", "rock paper scissors"),
           ("повторяй за мной", "mirror me"), ("отражай меня", "mirror me"), ("копируй меня", "copy me"),
           ("следуй за мной", "shadow me"), ("помаши", "wave"), ("палец вверх", "thumbs up"), ("кивни", "nod"),
           ("покачай головой", "shake your head"), ("покажи", "point"), ("укажи", "point"),
           ("посмотри на меня", "look at me"), ("смотри", "look"), ("дай мне", "give it to me"), ("отпусти", "let go"),
           ("возьми", "grip"), ("сожми", "grip"), ("аккуратно", "gently"), ("нежно", "gently"), ("празднуй", "celebrate"),
           ("пожми плечами", "shrug"), ("расслабься", "relax"), ("отдохни", "rest"), ("левой", "left"),
           ("налево", "left"), ("правой", "right"), ("направо", "right"), ("вверх", "up"), ("вниз", "down"),
           ("рукой", "hand"), ("привет", "hello"), ("что я держу", "what am i holding"), ("что ты видишь", "what do you see")],
    "ar": [("توقف عن تقليدي", "stop mirroring me"), ("حجر ورقة مقص", "rock paper scissors"),
           ("حجرة ورقة مقص", "rock paper scissors"), ("قلدني", "mirror me"), ("اتبعني", "shadow me"),
           ("انسخني", "copy me"), ("لوّح", "wave"), ("لوح", "wave"), ("ارفع إبهامك", "thumbs up"),
           ("أومئ", "nod"), ("هز رأسك", "shake your head"), ("أشر", "point"), ("انظر إلي", "look at me"),
           ("انظر", "look"), ("أعطني", "give it to me"), ("اتركه", "let go"), ("أمسك", "grip"), ("بلطف", "gently"),
           ("احتفل", "celebrate"), ("استرخ", "relax"), ("اليسرى", "left"), ("يسار", "left"), ("اليمنى", "right"),
           ("يمين", "right"), ("فوق", "up"), ("تحت", "down"), ("يدك", "hand"), ("مرحبا", "hello"),
           ("ماذا أحمل", "what am i holding"), ("ماذا ترى", "what do you see")],
}

# Rock paper scissors: the throw words, and "ready", "again" and "no" for the game
THROWS = {
    "en": {"rock": ["rock"], "paper": ["paper"], "scissors": ["scissors", "scissor"]},
    "es": {"rock": ["piedra"], "paper": ["papel"], "scissors": ["tijera", "tijeras"]},
    "fr": {"rock": ["pierre"], "paper": ["feuille", "papier"], "scissors": ["ciseaux"]},
    "de": {"rock": ["stein"], "paper": ["papier"], "scissors": ["schere"]},
    "it": {"rock": ["sasso", "pietra"], "paper": ["carta"], "scissors": ["forbice", "forbici"]},
    "pt": {"rock": ["pedra"], "paper": ["papel"], "scissors": ["tesoura"]},
    "hi": {"rock": ["पत्थर", "patthar"], "paper": ["कागज़", "कागज", "kagaz"], "scissors": ["कैंची", "kainchi"]},
    "zh": {"rock": ["石头"], "paper": ["布"], "scissors": ["剪刀"]},
    "ja": {"rock": ["グー", "ぐー", "石"], "paper": ["パー", "ぱー", "紙"], "scissors": ["チョキ", "ちょき", "はさみ"]},
    "ko": {"rock": ["바위", "주먹"], "paper": ["보", "보자기"], "scissors": ["가위"]},
    "ru": {"rock": ["камень"], "paper": ["бумага"], "scissors": ["ножницы"]},
    "ar": {"rock": ["حجر", "حجرة"], "paper": ["ورقة"], "scissors": ["مقص"]},
}
GAME_WORDS = {   # (ready, again, done)
    "en": (["ready", "go", "let's go", "start", "yes", "yeah", "ok", "okay"],
           ["again", "one more", "another", "yes", "yeah", "sure", "rematch", "play", "ok", "okay"],
           ["no", "nope", "done", "quit", "enough", "that's it", "i'm good"]),
    "es": (["listo", "lista", "vamos", "sí", "dale"], ["otra vez", "otra", "de nuevo", "sí", "revancha", "dale"],
           ["no", "basta", "ya está", "suficiente"]),
    "fr": (["prêt", "prête", "vas-y", "allez", "oui"], ["encore", "une autre", "revanche", "oui"],
           ["non", "ça suffit", "c'est bon", "fini"]),
    "de": (["bereit", "los", "ja"], ["nochmal", "noch einmal", "noch eine", "revanche", "ja"],
           ["nein", "genug", "fertig", "das reicht"]),
    "it": (["pronto", "pronta", "vai", "sì"], ["ancora", "un'altra", "rivincita", "sì"], ["no", "basta così", "finito"]),
    "pt": (["pronto", "pronta", "vamos", "sim", "bora"], ["de novo", "outra", "mais uma", "revanche", "sim"],
           ["não", "chega", "já deu", "acabou"]),
    "hi": (["तैयार", "चलो", "हाँ", "हां"], ["फिर से", "एक और", "दोबारा", "हाँ", "हां"], ["नहीं", "बस", "हो गया"]),
    "zh": (["准备好了", "好了", "开始", "来吧", "好"], ["再来", "再玩", "再一次", "好"], ["不", "不玩了", "够了", "算了"]),
    "ja": (["いいよ", "準備できた", "よし", "はい", "うん"], ["もう一回", "もういっかい", "もう一度", "はい", "うん"],
           ["いいえ", "もういい", "終わり", "おわり"]),
    "ko": (["준비됐어", "준비", "시작", "그래", "응"], ["한 번 더", "다시", "또", "그래", "응"], ["아니", "그만할래", "됐어", "끝"]),
    "ru": (["готов", "готова", "давай", "да", "поехали"], ["ещё", "еще", "ещё раз", "реванш", "да", "давай"],
           ["нет", "хватит", "всё", "достаточно"]),
    "ar": (["جاهز", "جاهزة", "هيا", "نعم", "يلا"], ["مرة أخرى", "مجددا", "نعم", "يلا"], ["لا", "كفى", "انتهينا"]),
}

# Piper voice per language: a male voice where there is one (Jarvis). Other Piper languages use their first voice.
VOICES = {
    "en": "en_GB-alan-medium", "es": "es_ES-davefx-medium", "fr": "fr_FR-tom-medium",
    "de": "de_DE-thorsten-medium", "it": "it_IT-riccardo-x_low", "pt": "pt_BR-faber-medium",
    "hi": "hi_IN-rohan-medium", "zh": "zh_CN-chaowen-medium", "ja": "ja_JP-hi_fi_captain-medium",
    "ko": "ko_KR-kss-medium", "ru": "ru_RU-dmitri-medium", "ar": "ar_JO-kareem-medium",
    "nl": "nl_NL-alex-medium", "pl": "pl_PL-darkman-medium", "tr": "tr_TR-dfki-medium", "sv": "sv_SE-nst-medium",
    "uk": "uk_UA-oleksa-high", "vi": "vi_VN-vais1000-medium", "id": "id_ID-news_tts-medium",
    "el": "el_GR-joy-medium", "he": "he_IL-saspeech-medium", "fa": "fa_IR-amir-medium", "cs": "cs_CZ-jirka-medium",
    "ro": "ro_RO-mihai-medium", "hu": "hu_HU-imre-medium", "th": "th_TH-tsync2-medium", "bn": "bn_BD-google-medium",
    "te": "te_IN-venkatesh-medium", "ml": "ml_IN-arjun-medium", "mr": "mr_IN-google-medium",
    "ur": "ur_PK-fasih-medium", "ne": "ne_NP-google-medium", "fi": "fi_FI-harri-medium",
    "da": "da_DK-talesyntese-medium", "no": "no_NO-talesyntese-medium", "ca": "ca_ES-upc_pau-x_low",
    "bg": "bg_BG-dimitar-medium", "sk": "sk_SK-lili-medium", "sl": "sl_SI-artur-medium",
    "sr": "sr_RS-serbski_institut-medium", "lt": "lt_LT-reginute1-medium", "lv": "lv_LV-aivars-medium",
    "et": "et_EE-news-medium", "ka": "ka_GE-natia-medium", "kk": "kk_KZ-issai-high", "sw": "sw_CD-lanfrica-medium",
    "is": "is_IS-steinn-medium", "cy": "cy_GB-gwryw_gogleddol-medium", "eu": "eu_ES-antton-medium",
    "hy": "hy_AM-gor-medium", "sq": "sq_AL-edon-medium", "lb": "lb_LU-marylux-medium",
}

_CJK = {"zh", "ja", "ko", "th"}
PUNCT = " ,.、。!！?？-—…'\"，：；،؟「」"
# Said before "Jarvis" in many languages, so they say nothing about which one
SHARED_GREETINGS = {"hey", "hei", "hej", "hi", "ey", "eh", "oh", "o", "ah", "a", "yo", "he", "hé", "hallo", "hello",
                    "ok", "okay", "so", "well", "um", "uh", "and", "dear", "my", "mr", "mister"}


def name(lang: str) -> str:
    return NAMES.get(lang, lang)


def voice_for(lang: str) -> str | None:
    return VOICES.get(lang)


def _has(text: str, word: str, lang: str) -> int:
    """Position of `word` in `text` as a whole word (any position for scripts without spaces), or -1."""
    if lang in _CJK or not word.isascii() and not any(c.isalpha() and c.isascii() for c in word):
        i = text.find(word)
        return i
    m = re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text)
    return m.start() if m else -1


def is_stop(text: str) -> bool:
    return bool(STOP_RE.search(text or ""))


def find_wake(text: str, strict: bool = True):
    """Find "Jarvis" with its wake words. Returns (command after the wake phrase, language hint or None), or None
    when "Jarvis" does not start the sentence: only greeting words may come before it ("wave at Jarvis" is not a
    wake). strict=False (speech) also lets up to two short unknown words through, for a greeting Whisper misheard
    ("o de Jarvis" for "oye Jarvis", "a cora de Jarvis" for "acorda Jarvis")."""
    t = (text or "").strip()
    m = JARVIS_RE.search(t)
    if not m:
        return None
    before, after = t[:m.start()].lower(), t[m.end():]
    hint, cut, rest = None, 0, before
    al = after.lower()
    for lang, words in WAKE_WORDS.items():
        for w in sorted(words, key=len, reverse=True):
            if _has(rest, w, lang) >= 0:
                hint = hint or lang
                rest = rest.replace(w, " ")
            i = _has(al[:16], w, lang)
            if i >= 0 and al[:i].strip(PUNCT) == "":
                hint, cut = hint or lang, max(cut, i + len(w))
    # Only greeting words may come before "Jarvis": "wave at Jarvis" is a sentence, not a wake phrase.
    unknown = [w for w in re.split(r"[\s" + re.escape(PUNCT) + r"]+", rest) if w and w not in SHARED_GREETINGS]
    if unknown and (strict or len(unknown) > 2 or any(len(w) > 4 for w in unknown)):
        return None
    command = after[cut:].lstrip(PUNCT + ":;").strip()
    return command, hint


def wake_language(text: str, whisper_lang: str | None = None, whisper_prob: float = 0.0,
                  previous: str = "en") -> str | None:
    """The language for this conversation, from a transcribed wake utterance (see the module notes)."""
    found = find_wake(text, strict=False)
    if found is None:
        return None
    _, hint = found
    if hint and hint != "en":
        return hint
    if whisper_lang and whisper_prob >= 0.5 and whisper_lang != "en":
        return whisper_lang
    if hint == "en" or whisper_lang == "en":
        return "en"
    return previous


def to_english(text: str, lang: str) -> str:
    """Swap a built-in language's command words for English ones, so the rule parser understands them."""
    t = (text or "").lower()
    for src, dst in sorted(COMMANDS.get(lang, []), key=lambda p: len(p[0]), reverse=True):
        t = t.replace(src.lower(), f" {dst} ")
    for rps, words in THROWS.get(lang, {}).items():
        for w in words:
            t = t.replace(w.lower(), f" {rps} ")
    return re.sub(r"\s+", " ", t).strip()


def throw_words(lang: str) -> dict:
    """word -> throw, for English plus the language spoken."""
    out = {}
    for code in ("en", lang):
        for throw, words in THROWS.get(code, {}).items():
            out.update({w.lower(): throw for w in words})
    return out


def game_words(lang: str):
    """(ready, again, done) word lists for English plus the language spoken."""
    en = GAME_WORDS["en"]
    other = GAME_WORDS.get(lang, ([], [], []))
    return tuple(a + b for a, b in zip(en, other))


def contains_any(text: str, words, lang: str) -> bool:
    t = (text or "").lower()
    return any(_has(t, w.lower(), lang) >= 0 for w in words)

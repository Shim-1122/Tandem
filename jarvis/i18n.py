"""What Jarvis says, in the language you spoke.

`tr(template, lang, **values)` looks the English template up in CATALOG and fills in the values. For a language
not in the catalog, a translator (the local language model, `OllamaBrain.translate`) translates the template
once and the result is cached in `models/translations.json`; with no translator the English is used.
Placeholders ({name}, {tally}...) stay as they are in every translation. The rock paper scissors count follows
each language's own chant where it has one (Japanese 最初はグー、じゃんけんぽん; Korean 안 내면 진다, 가위 바위 보;
German Schere, Stein, Papier), so the beats are not word-for-word translations.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

TEMPLATES = [
    "Hello.",                                                            # 0
    "Hello, {name}.",
    "Oh, it's you, {name}.",
    "Not good? Tell me what to change.",
    "Yes? I'm listening.",
    "You look a little down. Anything I can do?",                        # 5
    "I'll hold still.",
    "Nice {object}.",
    "Stopping.",
    "You're on! Show me a fist when you're ready.",
    "Let's just go.",                                                    # 10
    "Rock...",
    "Paper...",
    "Scissors...",
    "Shoot!",
    "Okay. Rock...",                                                     # 15
    "I didn't catch your throw. Say again to retry.",
    "{robot} beats {person}. I win! That's {tally}. Again?",
    "{person} beats {robot}. You got me. That's {tally}. Again?",
    "We both threw {throw}. Tie! That's {tally}. Again?",
    "{you} to {me}",                                                     # 20
    "{you} to {me}, {ties} tied",
    "Good game. You win the match, {you} to {me}.",
    "Good game. I win the match, {you} to {me}.",
    "Good game. It's a draw, {you} to {me}.",
    "Maybe later.",                                                      # 25
    "rock",
    "paper",
    "scissors",
    "Shadow mode off.",
    "Copying you, joint for joint.",                                     # 30
    "Mirroring you.",
    "Here you go.",
    "Woo!",
    "Relaxing.",
    "Okay.",                                                             # 35
    "Sorry, I didn't catch a command in that.",
    "Nothing yet.",
    "I can't see you right now.",
    "You look neutral to me.",
    "Nothing that I can see.",                                           # 40
]

CATALOG_LANGS = {
    "es": [
        "Hola.", "Hola, {name}.", "Ah, eres tú, {name}.", "¿No está bien? Dime qué cambio.", "¿Sí? Te escucho.",
        "Te veo un poco triste. ¿Puedo hacer algo?", "Me quedo quieto.", "Qué buen {object}.", "Me detengo.",
        "¡Acepto! Enséñame un puño cuando estés listo.", "Vamos ya.", "Piedra...", "Papel...", "Tijera...", "¡Ya!",
        "Vale. Piedra...", "No vi tu jugada. Di otra vez para repetir.",
        "{robot} gana a {person}. ¡Gané yo! Vamos {tally}. ¿Otra?",
        "{person} gana a {robot}. Me ganaste. Vamos {tally}. ¿Otra?",
        "Los dos sacamos {throw}. ¡Empate! Vamos {tally}. ¿Otra?", "{you} a {me}", "{you} a {me}, {ties} empates",
        "Buen juego. Ganas la partida, {you} a {me}.", "Buen juego. Gano yo la partida, {you} a {me}.",
        "Buen juego. Es un empate, {you} a {me}.", "Quizás luego.", "piedra", "papel", "tijera",
        "Modo sombra desactivado.", "Te copio, articulación por articulación.", "Te imito como un espejo.",
        "Aquí tienes.", "¡Bien!", "Me relajo.", "Vale.", "Perdona, no entendí ninguna orden.", "Nada todavía.",
        "Ahora no te veo.", "Te veo tranquilo.", "Nada que yo vea.",
    ],
    "fr": [
        "Bonjour.", "Bonjour, {name}.", "Ah, c'est toi, {name}.", "Pas bien ? Dis-moi quoi changer.",
        "Oui ? Je t'écoute.", "Tu as l'air un peu triste. Je peux faire quelque chose ?", "Je ne bouge plus.",
        "Joli {object}.", "J'arrête.", "Je relève le défi ! Montre-moi un poing quand tu es prêt.", "Allons-y.",
        "Pierre...", "Feuille...", "Ciseaux...", "Chifoumi !", "D'accord. Pierre...",
        "Je n'ai pas vu ton coup. Dis encore pour recommencer.",
        "{robot} bat {person}. J'ai gagné ! Ça fait {tally}. On rejoue ?",
        "{person} bat {robot}. Tu m'as eu. Ça fait {tally}. On rejoue ?",
        "On a joué {throw} tous les deux. Égalité ! Ça fait {tally}. On rejoue ?", "{you} à {me}",
        "{you} à {me}, {ties} égalités", "Bien joué. Tu gagnes la partie, {you} à {me}.",
        "Bien joué. Je gagne la partie, {you} à {me}.", "Bien joué. Match nul, {you} à {me}.", "Plus tard, peut-être.",
        "pierre", "feuille", "ciseaux", "Mode ombre désactivé.", "Je te copie, articulation par articulation.",
        "Je t'imite comme un miroir.", "Tiens.", "Youpi !", "Je me détends.", "D'accord.",
        "Désolé, je n'ai pas compris de commande.", "Rien pour l'instant.", "Je ne te vois pas en ce moment.",
        "Tu as l'air neutre.", "Rien que je puisse voir.",
    ],
    "de": [
        "Hallo.", "Hallo, {name}.", "Ach, du bist es, {name}.", "Nicht gut? Sag mir, was ich ändern soll.",
        "Ja? Ich höre.", "Du wirkst etwas niedergeschlagen. Kann ich etwas tun?", "Ich halte still.",
        "Schöner {object}.", "Ich stoppe.", "Abgemacht! Zeig mir eine Faust, wenn du bereit bist.", "Dann los.",
        "Schere...", "Stein...", "Papier...", "Los!", "Okay. Schere...",
        "Ich habe deinen Wurf nicht gesehen. Sag nochmal, um es zu wiederholen.",
        "{robot} schlägt {person}. Ich gewinne! Es steht {tally}. Nochmal?",
        "{person} schlägt {robot}. Du hast mich. Es steht {tally}. Nochmal?",
        "Wir haben beide {throw}. Unentschieden! Es steht {tally}. Nochmal?", "{you} zu {me}",
        "{you} zu {me}, {ties} unentschieden", "Gutes Spiel. Du gewinnst das Match, {you} zu {me}.",
        "Gutes Spiel. Ich gewinne das Match, {you} zu {me}.", "Gutes Spiel. Unentschieden, {you} zu {me}.",
        "Vielleicht später.", "Stein", "Papier", "Schere", "Schattenmodus aus.", "Ich kopiere dich, Gelenk für Gelenk.",
        "Ich spiegle dich.", "Bitte schön.", "Juhu!", "Ich entspanne mich.", "Okay.",
        "Entschuldigung, ich habe keinen Befehl verstanden.", "Noch nichts.", "Ich sehe dich gerade nicht.",
        "Du wirkst neutral.", "Nichts, was ich sehen kann.",
    ],
    "it": [
        "Ciao.", "Ciao, {name}.", "Ah, sei tu, {name}.", "Non va bene? Dimmi cosa cambiare.", "Sì? Ti ascolto.",
        "Ti vedo un po' giù. Posso fare qualcosa?", "Sto fermo.", "Bello, un {object}.", "Mi fermo.",
        "Ci sto! Mostrami un pugno quando sei pronto.", "Andiamo.", "Sasso...", "Carta...", "Forbice...", "Via!",
        "Va bene. Sasso...", "Non ho visto la tua mossa. Di' ancora per riprovare.",
        "{robot} batte {person}. Ho vinto! Siamo {tally}. Ancora?",
        "{person} batte {robot}. Mi hai battuto. Siamo {tally}. Ancora?",
        "Abbiamo fatto entrambi {throw}. Pareggio! Siamo {tally}. Ancora?", "{you} a {me}",
        "{you} a {me}, {ties} pareggi", "Bella partita. Vinci tu, {you} a {me}.", "Bella partita. Vinco io, {you} a {me}.",
        "Bella partita. È un pareggio, {you} a {me}.", "Magari più tardi.", "sasso", "carta", "forbice",
        "Modalità ombra disattivata.", "Ti copio, articolazione per articolazione.", "Ti imito come uno specchio.",
        "Ecco a te.", "Evviva!", "Mi rilasso.", "Va bene.", "Scusa, non ho capito nessun comando.", "Ancora niente.",
        "Adesso non ti vedo.", "Mi sembri tranquillo.", "Niente che io possa vedere.",
    ],
    "pt": [
        "Olá.", "Olá, {name}.", "Ah, é você, {name}.", "Não gostou? Diga o que mudar.", "Sim? Estou ouvindo.",
        "Você parece meio triste. Posso ajudar?", "Vou ficar parado.", "Que {object} legal.", "Parando.",
        "Topo! Mostre um punho quando estiver pronto.", "Vamos logo.", "Pedra...", "Papel...", "Tesoura...", "Já!",
        "Certo. Pedra...", "Não vi sua jogada. Diga de novo para repetir.",
        "{robot} ganha de {person}. Ganhei! Está {tally}. De novo?",
        "{person} ganha de {robot}. Você me pegou. Está {tally}. De novo?",
        "Nós dois jogamos {throw}. Empate! Está {tally}. De novo?", "{you} a {me}", "{you} a {me}, {ties} empates",
        "Bom jogo. Você venceu a partida, {you} a {me}.", "Bom jogo. Eu venci a partida, {you} a {me}.",
        "Bom jogo. Deu empate, {you} a {me}.", "Talvez depois.", "pedra", "papel", "tesoura",
        "Modo sombra desligado.", "Copiando você, articulação por articulação.", "Imitando você como um espelho.",
        "Aqui está.", "Uhu!", "Relaxando.", "Certo.", "Desculpe, não entendi nenhum comando.", "Nada ainda.",
        "Não estou te vendo agora.", "Você parece tranquilo.", "Nada que eu consiga ver.",
    ],
    "hi": [
        "नमस्ते।", "नमस्ते, {name}।", "अरे, आप हैं, {name}।", "ठीक नहीं? बताइए क्या बदलूँ।", "हाँ? मैं सुन रहा हूँ।",
        "आप थोड़े उदास लग रहे हैं। क्या मैं कुछ कर सकता हूँ?", "मैं स्थिर रहूँगा।", "बढ़िया {object}।", "रुक रहा हूँ।",
        "मंज़ूर! तैयार हों तो मुट्ठी दिखाइए।", "चलिए शुरू करते हैं।", "पत्थर...", "कागज़...", "कैंची...", "चलो!",
        "ठीक है। पत्थर...", "मैंने आपकी चाल नहीं देखी। दोबारा खेलने के लिए फिर से कहिए।",
        "{robot} ने {person} को हराया। मैं जीता! स्कोर {tally}। फिर से?",
        "{person} ने {robot} को हराया। आप जीत गए। स्कोर {tally}। फिर से?",
        "हम दोनों ने {throw} चुना। बराबरी! स्कोर {tally}। फिर से?", "{you}-{me}", "{you}-{me}, {ties} बराबर",
        "बढ़िया खेल। आप मैच जीत गए, {you}-{me}।", "बढ़िया खेल। मैं मैच जीत गया, {you}-{me}।",
        "बढ़िया खेल। मैच बराबरी पर रहा, {you}-{me}।", "शायद बाद में।", "पत्थर", "कागज़", "कैंची",
        "छाया मोड बंद।", "आपकी नकल कर रहा हूँ, हर जोड़ के साथ।", "आपको आईने की तरह दोहरा रहा हूँ।", "यह लीजिए।",
        "वाह!", "आराम कर रहा हूँ।", "ठीक है।", "माफ़ कीजिए, मुझे कोई आदेश समझ नहीं आया।", "अभी कुछ नहीं।",
        "मैं अभी आपको नहीं देख पा रहा।", "आप सामान्य लग रहे हैं।", "मुझे कुछ नहीं दिख रहा।",
    ],
    "zh": [
        "你好。", "你好，{name}。", "哦，是你啊，{name}。", "不好吗？告诉我要改什么。", "嗯？我在听。",
        "你看起来有点难过。我能做点什么吗？", "我不动了。", "这个{object}真不错。", "停下了。",
        "来吧！准备好了就握个拳头给我看。", "那就开始吧。", "石头……", "剪刀……", "布……", "出！", "好。石头……",
        "我没看到你出的什么。说“再来”重新开始。", "{robot}赢{person}。我赢了！现在{tally}。再来？",
        "{person}赢{robot}。你赢了。现在{tally}。再来？", "我们都出了{throw}。平局！现在{tally}。再来？",
        "{you}比{me}", "{you}比{me}，平局{ties}次", "好比赛。你赢了，{you}比{me}。", "好比赛。我赢了，{you}比{me}。",
        "好比赛。打平了，{you}比{me}。", "下次吧。", "石头", "布", "剪刀", "影子模式关闭。", "逐个关节复制你的动作。",
        "像镜子一样模仿你。", "给你。", "耶！", "放松一下。", "好的。", "抱歉，我没听出指令。", "还没有。",
        "我现在看不到你。", "你看起来很平静。", "我什么也没看到。",
    ],
    "ja": [
        "こんにちは。", "こんにちは、{name}さん。", "あ、{name}さんですね。", "だめですか？何を変えればいいか教えてください。",
        "はい？聞いています。", "少し元気がないようですね。何かできることはありますか？", "じっとしています。",
        "いい{object}ですね。", "止まります。", "受けて立ちます！準備ができたらグーを見せてください。", "始めましょう。",
        "最初は...", "グー...", "じゃんけん...", "ぽん！", "はい。最初は...", "手が見えませんでした。「もう一回」と言ってください。",
        "{robot}は{person}に勝ち。私の勝ち！{tally}です。もう一回？", "{person}は{robot}に勝ち。あなたの勝ち。{tally}です。もう一回？",
        "二人とも{throw}。あいこ！{tally}です。もう一回？", "{you}対{me}", "{you}対{me}、あいこ{ties}回",
        "いい勝負でした。あなたの勝ちです、{you}対{me}。", "いい勝負でした。私の勝ちです、{you}対{me}。",
        "いい勝負でした。引き分けです、{you}対{me}。", "また今度。", "グー", "パー", "チョキ", "シャドーモードを終了します。",
        "関節ごとにまねします。", "鏡のようにまねします。", "どうぞ。", "やった！", "リラックスします。", "はい。",
        "すみません、命令が聞き取れませんでした。", "まだ何もありません。", "今はあなたが見えません。", "落ち着いて見えます。",
        "何も見えません。",
    ],
    "ko": [
        "안녕하세요.", "안녕하세요, {name}님.", "아, {name}님이군요.", "별로예요? 뭘 바꿀지 말해 주세요.", "네? 듣고 있어요.",
        "조금 우울해 보여요. 제가 도울 일이 있을까요?", "가만히 있을게요.", "멋진 {object}네요.", "멈출게요.",
        "좋아요! 준비되면 주먹을 보여 주세요.", "그냥 시작할게요.", "안 내면 진다...", "가위...", "바위...", "보!", "좋아요. 안 내면 진다...",
        "손을 못 봤어요. 다시 하려면 '다시'라고 말해 주세요.", "{robot}이(가) {person}을(를) 이겨요. 제가 이겼어요! {tally}예요. 한 번 더?",
        "{person}이(가) {robot}을(를) 이겨요. 졌어요. {tally}예요. 한 번 더?", "둘 다 {throw}! 비겼어요! {tally}예요. 한 번 더?",
        "{you} 대 {me}", "{you} 대 {me}, 무승부 {ties}번", "좋은 게임이었어요. 당신이 이겼어요, {you} 대 {me}.",
        "좋은 게임이었어요. 제가 이겼어요, {you} 대 {me}.", "좋은 게임이었어요. 무승부예요, {you} 대 {me}.", "나중에 해요.",
        "바위", "보", "가위", "그림자 모드를 껐어요.", "관절 하나하나 따라 할게요.", "거울처럼 따라 할게요.", "여기 있어요.",
        "와!", "쉴게요.", "네.", "죄송해요, 명령을 알아듣지 못했어요.", "아직 없어요.", "지금은 당신이 안 보여요.",
        "차분해 보여요.", "아무것도 안 보여요.",
    ],
    "ru": [
        "Привет.", "Привет, {name}.", "А, это ты, {name}.", "Не так? Скажи, что изменить.", "Да? Я слушаю.",
        "Ты выглядишь немного грустным. Могу чем-то помочь?", "Замираю.", "Отличный {object}.", "Останавливаюсь.",
        "Принято! Покажи кулак, когда будешь готов.", "Тогда начинаем.", "Камень...", "Ножницы...", "Бумага...", "Раз!",
        "Хорошо. Камень...", "Я не увидел твой ход. Скажи «ещё», чтобы повторить.",
        "{robot} бьёт {person}. Я выиграл! Счёт {tally}. Ещё?", "{person} бьёт {robot}. Ты меня обыграл. Счёт {tally}. Ещё?",
        "Мы оба выбрали {throw}. Ничья! Счёт {tally}. Ещё?", "{you}:{me}", "{you}:{me}, ничьих {ties}",
        "Хорошая игра. Ты выиграл матч, {you}:{me}.", "Хорошая игра. Я выиграл матч, {you}:{me}.",
        "Хорошая игра. Ничья, {you}:{me}.", "Может, позже.", "камень", "бумага", "ножницы", "Режим тени выключен.",
        "Копирую тебя, сустав за суставом.", "Отражаю тебя, как зеркало.", "Держи.", "Ура!", "Расслабляюсь.", "Хорошо.",
        "Извини, я не понял команду.", "Пока ничего.", "Сейчас я тебя не вижу.", "Ты выглядишь спокойным.",
        "Ничего не вижу.",
    ],
    "ar": [
        "مرحبا.", "مرحبا يا {name}.", "آه، إنه أنت يا {name}.", "ليس جيدا؟ قل لي ماذا أغير.", "نعم؟ أنا أستمع.",
        "تبدو حزينا قليلا. هل يمكنني فعل شيء؟", "سأبقى ثابتا.", "{object} جميل.", "أتوقف.",
        "موافق! أرني قبضتك عندما تكون جاهزا.", "لنبدأ إذن.", "حجر...", "ورقة...", "مقص...", "الآن!", "حسنا. حجر...",
        "لم أرَ رميتك. قل مرة أخرى للإعادة.", "{robot} يغلب {person}. فزت! النتيجة {tally}. مرة أخرى؟",
        "{person} يغلب {robot}. لقد غلبتني. النتيجة {tally}. مرة أخرى؟", "اخترنا كلانا {throw}. تعادل! النتيجة {tally}. مرة أخرى؟",
        "{you} مقابل {me}", "{you} مقابل {me}، و{ties} تعادل", "لعبة جميلة. فزت بالمباراة، {you} مقابل {me}.",
        "لعبة جميلة. فزت أنا بالمباراة، {you} مقابل {me}.", "لعبة جميلة. تعادلنا، {you} مقابل {me}.", "ربما لاحقا.",
        "حجر", "ورقة", "مقص", "تم إيقاف وضع الظل.", "أقلدك مفصلا بمفصل.", "أقلدك مثل المرآة.", "تفضل.", "رائع!",
        "أسترخي.", "حسنا.", "عذرا، لم أفهم أي أمر.", "لا شيء بعد.", "لا أراك الآن.", "تبدو هادئا.", "لا أرى شيئا.",
    ],
}
CATALOG = {lang: dict(zip(TEMPLATES, rows)) for lang, rows in CATALOG_LANGS.items()}


class Translator:
    """Translations for languages outside the catalog, made once by a model and kept on disk."""

    def __init__(self, translate_fn=None, path: str | Path = "models/translations.json"):
        self.translate_fn = translate_fn
        self.path = Path(path)
        self._lock = threading.Lock()
        try:
            self.cache = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.cache = {}

    def get(self, text: str, lang: str):
        with self._lock:
            hit = self.cache.get(lang, {}).get(text)
        if hit is not None or self.translate_fn is None:
            return hit
        try:
            out = self.translate_fn(text, lang)
        except Exception:
            return None
        if not out or set(_fields(out)) != set(_fields(text)):
            return None                        # a translation that lost a {placeholder} is not usable
        with self._lock:
            self.cache.setdefault(lang, {})[text] = out
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=1), encoding="utf-8")
            except OSError:
                pass
        return out


def _fields(s: str):
    import string
    return [f for _, f, _, _ in string.Formatter().parse(s) if f]


def tr(template: str, lang: str = "en", translator: Translator | None = None, **values) -> str:
    """The template in `lang` with the values filled in (English when no translation is known)."""
    text = template
    if lang and lang != "en":
        text = CATALOG.get(lang, {}).get(template) or (translator.get(template, lang) if translator else None) \
            or template
    try:
        return text.format(**values)
    except (KeyError, IndexError, ValueError):
        return template.format(**values)

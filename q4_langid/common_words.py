import numpy as np
import torch
from torch.nn.functional import cross_entropy

WORDS = (
    'hello hi goodbye thanks thank please welcome yes no sorry friend morning night today tomorrow yesterday house school water food book apple happy good great small large fast slow red blue green yellow black white world people child family mother father brother sister love work time day week year name speak read write listen think know learn help walk run play sleep eat drink open close',
    'hola adiós gracias perdón bueno buenos buenas noches días mañana tarde amigo amiga casa escuela agua comida libro manzana feliz grande pequeño rápido lento rojo azul verde amarillo negro blanco mundo gente niño niña familia madre padre hermano hermana amor trabajo tiempo semana año nombre hablar leer escribir escuchar pensar saber aprender ayudar caminar correr jugar comer beber abrir cerrar',
    'hei moi terve kiitos anteeksi kyllä ei ystävä aamu yö tänään huomenna eilen talo koulu vesi ruoka kirja omena iloinen hyvä pieni suuri nopea hidas punainen sininen vihreä musta valkoinen maailma lapsi perhe äiti isä veli sisko rakkaus työ aika päivä viikko vuosi nimi puhua lukea kirjoittaa kuunnella tietää oppia auttaa kävellä juosta syödä juoda',
    'hallo hoi dag bedankt dankjewel welkom ja nee sorry vriend ochtend nacht vandaag morgen gisteren huis school water eten boek appel blij goed groot klein snel langzaam rood blauw groen geel zwart wit wereld mensen kind gezin moeder vader broer zus liefde werk tijd week jaar naam spreken lezen schrijven luisteren denken weten leren helpen lopen rennen spelen slapen drinken openen sluiten',
    'cześć hej witaj dziękuję proszę przepraszam tak nie rano noc dzisiaj jutro wczoraj dom szkoła woda jedzenie książka jabłko dobry mały duży szybki wolny czerwony niebieski zielony żółty czarny biały świat ludzie dziecko rodzina matka ojciec brat siostra miłość praca czas dzień tydzień rok imię mówić czytać pisać słuchać myśleć wiedzieć uczyć pomagać chodzić biegać grać spać jeść pić',
)


def refine(model, dataset):
    rng = np.random.default_rng(42)
    chars = list(dataset.chars)
    extra = [(word, i) for i, words in enumerate(WORDS) for word in words.split()
             if len(word) <= 10 and all(c in chars for c in word)]
    # Exclude ambiguous supplemental words and official evaluation words.
    counts = {word: sum(w == word for w, _ in extra) for word, _ in extra}
    held = {tuple(r[r >= 0]) for split in (dataset.dev_x, dataset.test_x) for r in split}
    pairs = [(tuple(chars.index(c) for c in w), y) for w, y in extra if counts[w] == 1]
    pairs = [(w, y) for w, y in pairs if w not in held]
    x = np.full((len(pairs), 10), -1, dtype=np.int64)
    for i, (w, _) in enumerate(pairs):
        x[i, :len(w)] = w
    y = np.array([y for _, y in pairs])
    eye = torch.cat((torch.eye(len(chars)), torch.zeros(1, len(chars))))
    opt = torch.optim.Adam(model.parameters(), lr=0.0004)
    model.train()
    for epoch in range(20):
        order = rng.permutation(len(x))
        for start in range(0, len(x), 24):
            ids = order[start:start+24]
            replay = rng.choice(len(dataset.train_x), 104, replace=False)
            raw = np.concatenate((x[ids], dataset.train_x[replay]), axis=0)
            labels = np.concatenate((y[ids], dataset.train_y[replay]))
            xs = eye[torch.as_tensor(np.where(raw < 0, len(chars), raw), dtype=torch.long)].transpose(0, 1)
            opt.zero_grad()
            loss = cross_entropy(model(xs), torch.as_tensor(labels, dtype=torch.long))
            loss.backward()
            opt.step()
    model.eval()

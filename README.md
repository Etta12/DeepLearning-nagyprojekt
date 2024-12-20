# Image classification "in-the-wild" 

Description:

The goal of this project is to gather you own real-world image dataset with different types of objects in it and train an image classifier for solving it. Dataset examples: books vs shoes vs furniture, houseplants vs outdoor plants vs trees. Make sure that your dataset contains at least 3 classes, with at least 50 examples each (1 image per object). Make sure that the gathered images are diverse, with varied view angles, background and lighting. Select classes such that the task is not trivial to solve.

The main difficulty in this task will be the small dataset size, think through what we learned in class for solving it. Build an image classification pipeline for your dataset and train multiple networks on it, compare their performance. Inspect failure cases.

**Neurosünök**:
  - Wiederschitz Diána, Neptun kód: E0YKJT
  - Oroszki Marietta, Neptun kód: CZYYIF

## Projekt leírás
A Kaggle oldalán találtunk sakkfiguráról készített adatokat, innen jött az ötlet hogy egy olyan osztályozót szeretnénk létrehozni, ami 3 sakkfigurát: a bástyát, futót és huszárt meg tudja különböztetni egymástól. A feladat leírásának eleget téve készítettünk egy összesen 150 képből álló adathalmazt, ahol minden figuráról összesen 50 kép készült. A képek készítése során 5 különböző sakkészlet fekete és fehér bábuit használtuk fel,  és a változatos adathalmaz elérése érdekében igyekeztünk minél különbözőbb szögekből, pozíciókból és hátterekkel lefotózni őket. Végül az elkészült képeket véletlenszerű sorrendben összesítettük, majd megírtuk a címkéket tartalmazó Excel fájlt is, ahol a címkék az {1,2,3} halmazból kerülnek ki, a jelentésük pedig sorban: bástya, huszár, futó.
A későbbi kísérletezésekhez a már említett, Kaggle oldalán talált képes adatot is letöltöttük egy Jupyter Notebook segítségével.

Az adathalmazt méreténél fogva nem tudtuk feltölteni ide, de a következő Google Drive linket elérhető: 
https://drive.google.com/drive/folders/1r7_hwPG0_pSRdOsk5v7ltxyH0n5JP01l?usp=sharing

A tartalma: 
  - `images`: Ez a mappa tartalmazza a 150 képet.
  - `labels.xlsx`: Tartalmazza a 150 kép mindegyikéhez a címkéket.

## Fájlok

- `final.ipynb`: A végső feladatmegoldást tartalmazó Jupyter notebook.
- `Dokumentacio.pdf`: A végső feladatmegoldáshoz tartozó dokumentáció.

------------------------------------------------

- `data_acquisition.ipynb`: A Kaggle adathalmaz letöltéséhez használt Jupyter notebook, az adat végül nem került további felhasználásra.
- `data_preparation.py`: Az adatok előkészítéséhez használt Python script. (1. mérföldkő)
- `data_preparation_with_baseline_model.py`: Az adatok előkészítéséhez és a az első, kezdetleges modell kiértékeléséhez használt Python script. (2. mérföldkő)
- `requirements.txt`: A 2. mérföldkőhöz szükséges Python csomagok és függőségek.
- `Dockerfile`: A konténerizált környezet definiálása a futtatáshoz (2. mérföldkő).
- `Proba.ipynb`: Ketten dolgoztunk a projekten, ezért könnybnek találtuk, ha ugyanabból a vázból két féle képpen is elindulunk és ezáltal több féle eredmény várt elérhetővé.

## Használat a 2. mérföldkőhöz:
1. A data_preparation_with_baseline_model.py fájl Docker környezetbeli futtatásához szükséges egy mappába letölteni magát a .py fájlt, a Drive linken elérhető adatokat (images mappa és tartalma, labels.xlsx), illetve a requirements.txt-t és a Dockerfile-t.
2. Ezután szükséges a mappába navigálva az alábbi sorokat parancssorból futtatni:
    ```bash
    docker build -t chess_project .
    docker run --rm chess_project

## Használat a final.ipynb-hez:
A Drive linken lévő adatokat fel szükséges tölteni a saját drivera, majd a megnyitott Jupyter notebookban szükséges a megfelelő fülnél megadni az images mappa és a labels.txt pontos elérési útját.

## Megjegyzés
A végső feladatmegoldás nem konténerizált környezetben zajlott, sajnos nem tudtuk megoldani a GPU használatát, így Colabban dolgoztunk.

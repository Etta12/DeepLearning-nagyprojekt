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
- `data_acquisition.ipynb`: A saját és a Kaggle adathalmaz letöltéséhez használt Jupyter notebook.
- `data_preparation.py`: Az adatok előkészítéséhez használt Jupyter notebook.
- `requirements.txt`: A projekthez szükséges Python csomagok és függőségek.
- `Dockerfile`: A konténerizált környezet definiálása a futtatáshoz.

## Használat és magyarázat
0. A data_acquisition.ipynb fájl a Kaggle oldaláról képes elérni a már említett adathalmazt, de egyelőre még nincs további felhasználása ennek a fájlnak, illetve az adatnak.  
1. A data_preparation.py fájl Docker környezetbeli futtatásához szükséges egy mappába letölteni magát a .py fájlt, a Drive linken elérhető adaatokat (images mappa és tartalma, labels.xlsx), illetve a requirements.txt-t és a Dockerfile-t.
2. Ezután szükséges a mappába navigálva az alábbi sorokat parancssorból futtatni:
    ```bash
    docker build -t chess_project .
    docker run --rm chess_project

## Kapcsolódó munkák
https://arxiv.org/abs/1512.03385
https://www.sciencedirect.com/science/article/pii/S0031320316303922

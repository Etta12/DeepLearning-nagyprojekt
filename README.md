# Image classification "in-the-wild" 

Description:

The goal of this project is to gather you own real-world image dataset with different types of objects in it and train an image classifier for solving it. Dataset examples: books vs shoes vs furniture, houseplants vs outdoor plants vs trees. Make sure that your dataset contains at least 3 classes, with at least 50 examples each (1 image per object). Make sure that the gathered images are diverse, with varied view angles, background and lighting. Select classes such that the task is not trivial to solve.

The main difficulty in this task will be the small dataset size, think through what we learned in class for solving it. Build an image classification pipeline for your dataset and train multiple networks on it, compare their performance. Inspect failure cases.

**Neurosünök**:
  - Wiederschitz Diána, Neptun kód: E0YKJT
  - Oroszki Marietta, Neptun kód: CZYYIF

## Projekt leírás
A Kaggle oldalán találtunk sakkfiguráról készített adatokat, innen jött az ötlet hogy egy olyan osztályozót szeretnénk létrehozni, ami 3 sakkfigurát: a bástyát, futót és huszárt meg tudja különböztetni egymástól. A feladat leírásának eleget téve készítettünk egy összesen 150 képből álló adathalmazt, ahol minden figuráról összesen 50 kép készült. A képek készítése során 5 különböző sakkészlet bábuit használtuk fel,  és a változatos adathalmaz elérése érdekében igyekeztünk minél különbözőbb szögekből, pozíciókból és hátterekkel lefotózni őket. 
A későbbi kísérletezésekhez a már említett, Kaggle oldalán talált képes adatot is letöltöttük Python kód segítségével.

## Fájlok
- `images`: Tárolja a 150 db általunk előállított képekből álló tanulóhalmazt.
- `labels.xlsx`: Tárolja a 150 képhez tartozó címkéket. 
- `requirements.txt`: A projekthez szükséges Python csomagok és függőségek.
- `Dockerfile`: A konténerizált környezet definiálása a futtatáshoz.
- `data_acquisition.ipynb`: A Kaggle adathalmaz letöltéséhez használt Jupyter notebook.
- `data_preparation.ipynb`: Az adatok előkészítéséhez használt Jupyter notebook.

## Használat
1. **Környezet beállítása**
2. ...

## Kapcsolódó munkák
https://arxiv.org/abs/1512.03385
https://www.sciencedirect.com/science/article/pii/S0031320316303922

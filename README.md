# druckteil-optimierung

Rechen-Repo für die Topologie-Optimierung von 3D-Druckteilen (Resin): Lager
und Lasten festlegen, Spannungen per Voxel-FEM rechnen, Schutzbereiche massiv
halten, nur den Lastpfad behalten. Das Werkzeug liegt in
`werkzeuge/topopt.py`, je Teil ein Ordner mit seinen Skripten.

Gesteuert wird das Repo aus einem anderen Repo heraus: Werkzeug und
Teil-Skripte werden hierher gespiegelt, dann startet ein `workflow_dispatch`
den Workflow **Optimieren**. Jeder Eintrag in `laeufe` rechnet auf einem
eigenen Runner, alle gleichzeitig. Die Ergebnisse (STL, Bilder, Berichte,
Protokoll) hängen 14 Tage als Artefakt am Lauf.

Von Hand starten:

```sh
gh workflow run optimieren.yml -f teil=balkenschuh-9mm \
  -f laeufe='["wand --vol 0.5", "holz --vol 0.35"]'
```

Die Rechnung ist eine Näherung (lineare Elastizität, Quader-Elemente,
Spannungen an Elementmitten) und ersetzt keinen Festigkeitsnachweis.

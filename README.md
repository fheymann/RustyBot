Hinweis:

Während der Scrims und Finalrunden läuft dein Bot in einem Docker-Container ohne Netzwerk-Zugriff.
Den braucht cargo aber, um fehlende Bibliotheken zu holen. Du musst sie also deinem Quelltext beilegen,
so dass Cargo ohne Netzwerk-Zugriff klarkommt. Führe dafür einfach diesen Befehl aus:

```
cargo vendor vendor
```

Dann bekommst du ein Verzeichnis `vendor`, das du committen musst, und dann sollte dein Bot auch auf dem Server korrekt laufen. 

Dieses Rust-Template wurde von Matthias Fuchs geschrieben, danke dafür!

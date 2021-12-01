Am rulat si testat codul doar pe sisteme cu Linux, din pacate n-am avut cum sa-l testez pe Windows, deci nu stiu daca merge.
Toate informatiile de mai jos le puteti citi si in README.md si pe pagina mea de GitHub : https://github.com/andrei-brihac/sudoku-cv
(cred ca sunt mai lizibile pe pagina de github)

Pentru a-l rula puteti executa src/main.py cu un IDE (eu am folosit VSCode) sau puteti rula din root directory-ul proiectului comanda:

python3 src/main.py

Numele folderelor de intrare si iesire le puteti modifica din src/utils.py, variabilele cwd, input_path si output_path.
De asemenea, in src/main.py in lista types sunt numele folderelor care contin imagini pentru fiecare task si influenteaza si ce task se ruleaza.
    Ex: eu am avut numele classic si jigsaw pentru foldere, daca sunt diferite puteti modifica si ar trebui sa mearga.

Observatie: folder-ul de output trebuie deja sa fie creat, programul meu nu-l creeaza automat

Script-ul dvs. de notare ar trebui sa mearga daca modificati path-urile.

Sper sa fie de ajuns pentru punctul din oficiu :D 
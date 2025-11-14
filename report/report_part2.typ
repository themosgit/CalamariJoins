#set text(font: "New Computer Modern", lang: "el")


#set page(
  paper: "a4",
  margin: (top: 2.5cm, bottom: 2.5cm, left: 2.5cm, right: 2.5cm),
)

#set heading(numbering: "1.")


#show heading: it => {
  set text(weight: "bold")
  v(1em)
  // line(length: 100%)
  align(left)[#it.body]
  v(0.35em)
}

#align(center)[#text(22pt)[Παραδοτέο 2]]

#align(center)[#text(16pt)[Μάθημα: Ανάπτυξη Λογισμικού για Πληροφοριακά Συστήματα]]

#v(3em)
#align(center)[#text(14pt)[Μέλη Ομάδας:]]
#align(center)[#text(12pt)[Ραμαντάν Κονόμι - ΑΜ: 1115201800281]]
#align(center)[#text(12pt)[Θεμιστοκλής Παπαθεοφάνους - ΑΜ: 1115202100227]]
#align(center)[#text(12pt)[Μάριος Γιαννόπουλος - ΑΜ: 1115202000032]]

#outline(
  title: "Πίνακας Περιεχομένων",
  depth: 2,
  indent: auto,
)
#pagebreak()
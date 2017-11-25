from neurons import letters

fIn = open("moby_dick.txt", "r")
fOut = open("moby_dick_cleaned.txt", "w")

orig = fIn.read()

oneline = orig.replace("\r","").replace("\n","").replace("“","\"").replace("”","\"").replace("’","'").replace("-","-").lower()

cleaned = "".join([(x if x in letters else "_") for x in oneline])

while cleaned.find("  ") != -1:
  cleaned = cleaned.replace("  "," ")

fOut.write(cleaned)

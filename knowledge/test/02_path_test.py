from pathlib import Path

print(Path(__file__))
print(Path(__file__).parent.parent)
print(Path(__file__).parent)
print(Path(__file__).parents[0])
print(Path(__file__).parents[1])

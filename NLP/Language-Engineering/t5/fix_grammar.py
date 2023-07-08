import happytransformer
from happytransformer import HappyTextToText, TTSettings

happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")

args = TTSettings(num_beams=5, min_length=1)

# Add the prefix "grammar: " before each input
result = happy_tt.generate_text("grammar: she decide to use her drones to strike londoner city. however, when the cia investigates hell'd be hiding somewhere in the city, they want cross to help them find hellinger before he gets involved again.", args=args)

print(result.text)

#This is just a test, we never actually used this code.
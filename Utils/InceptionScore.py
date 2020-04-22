

class InceptionScore():

# Il modello pre-trainato, per ogni immagine, restituisce una valore di
# probabilità compreso tra 0 e 1 che indica la probabilità della slice di
# appartenere ad un paziente adaptive. Quindi la probabilità che la slice
# appartenga ad un paziente non adaptive è 1-sigmoid_output

    def compute_inception_score(self):
        # Calcolo p(y):
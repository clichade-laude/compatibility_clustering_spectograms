import numpy as np
import pickle

def generate_backdoor_poison(poison_size=1, img_size=32, seed=100):
    #seed = 100 # scenario 1
    #seed = 1000 # scenario 2
    #seed = 10000 # scenario 3
    if seed is not None:
        np.random.seed(seed)
    else:
        seed = np.random.randint(50000000)
        np.random.seed(seed)

    pairs = [(0, 2)]
    poison_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.]

    for source, target in pairs:
        position = np.random.randint(img_size-poison_size, size=(2,))
        color = np.random.randint(255, size=(3,))
        for f in poison_levels:
            ds_name = f"database/backdoor/backdoor_{source}-{target}_{f}_{poison_size}-{img_size}.pickle"
            params = {"size": poison_size,
                      "position": position,
                      "color": color,
                      "fraction_poisoned": f,
                      "seed": seed + int(f * 100) + source * 10 + target,
                      "source": source,
                      "target": target}
            with open(ds_name, 'wb') as f:
                pickle.dump(params, f)

if __name__ == "__main__":
    generate_backdoor_poison()


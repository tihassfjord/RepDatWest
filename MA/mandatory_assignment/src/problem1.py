import sys
import numpy as np
np.random.seed(1)
sys.path.append('code/')
sys.path.append('data/')
import gradient_checks
import cnn
from HE_mnist_loader import load_mnist, plot_mnist_results

print("******************\nSTARTING PROBLEM 1\n******************")
print("In order to validate your implementation of the backward pass we run a \
gradient check for the layer implementations. Note, that an relative error of less than 1e-6 \
most likely indicates that the gradient is correct.")

gradient_checks.check_gradient_conv()
gradient_checks.check_gradient_pool()
gradient_checks.check_gradient_relu()
gradient_checks.check_gradient_fc()

input("If the gradients look ok, press Enter to continue training the model...")

x_tr, y_tr, x_te, y_te = load_mnist(plot=True)

BS = 500

model = cnn.CNN()
model.train(x_tr, y_tr, 100, BS)

out = model.test(x_te)
pred = np.argmax(out, axis=1)

plot_mnist_results(x_te, pred, y_te)
print("ACC: ", np.sum(pred == y_te)/float(500))

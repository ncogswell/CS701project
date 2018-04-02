## Casey Astiz & Nick Cogswell
## CS701 project
## Spring 2018

## Skeleton Code for initial implementation
from sklearn.neural_network import MLPClassifier
from PIL import Image

## Reference: http://scikit-learn.org/stable/modules/neural_networks_supervised.html

#basic set up for the neural net

def main():
    """Main function for basic implementation"""
    ##data is going to come from recordings from the car -> openCV

    # data = [[0., 0.], [1., 1.]]
    # ##training_data = [80% of data]
    # labels = [0, 1, 2, 3]
    # ## representing 4 basic states (stop, forward, right, left)
    #
    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
    #                      hidden_layer_sizes=(5, 2), random_state=1)
    # clf.fit(data, labels) ## fit on train
    # return clf.predict([[2., 2.], [-1., -2.]]) ## predict on test data

    im = Image.open('RyanZotti/arrow_key_images/LeftArrow.tif')
    im.show()

main()

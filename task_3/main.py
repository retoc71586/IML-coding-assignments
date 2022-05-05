import utils_manzo
import utils_pelt
import classificationNet


def main():
    # unzipping
    utils_manzo.unzip('food.zip')

    # features extraction
    features = utils_pelt.backbone()

    # load classification net
    net = classificationNet.ClassificationNet().double()
    utils_manzo.trainModel(net, features)


if __name__ == '__main__':
    main()

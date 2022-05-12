import classificationNetTF
import classificationNetTorch
import utils_manzo
import utils_pelt


def main():
    train = True
    torch = False

    # unzipping
    utils_manzo.unzip('food.zip')

    if torch:
        # features extraction
        features = utils_pelt.backbone()

        # load classification net
        net = classificationNetTorch.ClassificationNet().double()
        # train
        if train:
            print('Training mode')
            utils_manzo.trainModelTorch(net, features)
        # test
        if not train:
            print('Eval mode')
            utils_manzo.evaluateModelTorch(net, features)

    if not torch:
        utils_manzo.pipeline_tensorflow()


if __name__ == '__main__':
    main()

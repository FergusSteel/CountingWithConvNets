from torchdataset import *
from convert_to_npz import show_density_map
from capsnet import CapsNetBasic, SegCaps, CCCaps, SegCapsOld, ReconstructionNet
from pilot_utils import TorchUNetModel
import torch
from capstrain import AverageMeter, compute_acc
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_test(capsmodel, cnn, dataset):
    caps_results = {
        "pc_err": [AverageMeter() for i in range(10)],
        "tot_err": AverageMeter(),
    }
    cnn_results = {
        "pc_err": [AverageMeter() for i in range(10)],
        "tot_err": AverageMeter(),
    }

    print("Running Test")
    for i in tqdm(range(1000)):
        input_dict = next(dataset)
        img = input_dict['image'].to(device)
        ground_truth = input_dict["dmap"]
        capsout = capsmodel(img.unsqueeze(0).float().to(device))[0]
        cnnout = cnn(img.unsqueeze(0).float().to(device))[0]

        caps_results["tot_err"].update(abs(sum(sum(sum(ground_truth.cpu().detach().numpy()[0]))) / 100 - sum(sum(sum(capsout.cpu().detach().numpy()[0]))) / 100))
        cnn_results["tot_err"].update(abs(sum(sum(sum(ground_truth.cpu().detach().numpy()[0]))) / 100 - sum(sum(sum(cnnout.cpu().detach().numpy()[0]))) / 100))

        for digit_class in range(10):
            caps_results["pc_err"][digit_class].update(compute_acc(capsout.cpu().detach().numpy(), ground_truth.cpu().detach().numpy(), digit_class))
            cnn_results["pc_err"][digit_class].update(compute_acc(cnnout.cpu().detach().numpy(), ground_truth.cpu().detach().numpy(), digit_class))

    return caps_results, cnn_results    


if "__main__" == __name__:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    CAPSWEIGHTS='segcaps_100Sigma2Good.pkl'
    BASELINEWEIGHTS='UNET_WEIGHTS.pkl'

    baseline = iter(DataLoader(SpreadMNISTDataset(int(1000), train=False, path="baseline"), batch_size=1, shuffle=False))

    fifteen_degree = iter(DataLoader(SpreadMNISTDataset(int(1000), train=False, path="fifteen_degrees"), batch_size=1, shuffle=False))
    thirty_degree = iter(DataLoader(SpreadMNISTDataset(int(1000), train=False, path="thirty_degrees"), batch_size=1, shuffle=False))
    forty_five_degree = iter(DataLoader(SpreadMNISTDataset(int(1000), train=False, path="fortyfive_degrees"), batch_size=1, shuffle=False))
    twentyfive_scale = iter(DataLoader(SpreadMNISTDataset(int(1000), train=False, path="twentyfive_perc"), batch_size=1, shuffle=False))
    fifty_scale = iter(DataLoader(SpreadMNISTDataset(int(1000), train=False, path="fifty_perc"), batch_size=1, shuffle=False))

    capsmodel = CapsNetBasic(10).to(device)
    cnn = TorchUNetModel(10).to(device)

    capsmodel.load_state_dict(torch.load(CAPSWEIGHTS))
    cnn.load_state_dict(torch.load(BASELINEWEIGHTS))

    print("ROTATIONAL VARIANCE TESTS")
    print("-"*50)

    zero_caps, zero_cnn = run_test(capsmodel, cnn, baseline)
    print("BASELINE RESULTS")
    print("CAPS RESULTS")
    print(f"Average Error: {zero_caps['tot_err'].avg}")
    print(f"Per Class Error: {[(dig, err.avg) for dig, err in enumerate(zero_caps['pc_err'])]}")

    print("CNN RESULTS")
    print(f"Average Error: {zero_cnn['tot_err'].avg}")
    print(f"Per Class Error: {[(dig, err.avg) for dig, err in enumerate(zero_cnn['pc_err'])]}")

    fifteen_caps, fifteen_cnn = run_test(capsmodel, cnn, fifteen_degree)
    print("FIFTEEN DEGREE RESULTS")
    print("CAPS RESULTS")
    print(f"Average Error: {fifteen_caps['tot_err'].avg}")
    print(f"Per Class Error: {[(dig, err.avg) for dig, err in enumerate(fifteen_caps['pc_err'])]}")

    print("CNN RESULTS")
    print(f"Average Error: {fifteen_cnn['tot_err'].avg}")
    print(f"Per Class Error: {[(dig, err.avg) for dig, err in enumerate(fifteen_cnn['pc_err'])]}")

    thirty_caps, thirty_cnn = run_test(capsmodel, cnn, thirty_degree)
    print()
    print("-"*50)
    print("THIRTY DEGREE RESULTS")
    print("CAPS RESULTS")
    print(f"Average Error: {thirty_caps['tot_err'].avg}")
    print(f"Per Class Error: {[(dig, err.avg) for dig, err in enumerate(thirty_caps['pc_err'])]}")

    print("CNN RESULTS")
    print(f"Average Error: {thirty_cnn['tot_err'].avg}")
    print(f"Per Class Error: {[(dig, err.avg) for dig, err in enumerate(thirty_cnn['pc_err'])]}")

    fortyfive_caps, fortyfive_cnn = run_test(capsmodel, cnn, forty_five_degree)
    print()
    print("-"*50)
    print("CAPS RESULTS")
    print("FORTY-FIVE DEGREE RESULTS")
    print(f"Average Error: {fortyfive_caps['tot_err'].avg}")
    print(f"Per Class Error: {[(dig, err.avg) for dig, err in enumerate(fortyfive_caps['pc_err'])]}")
    print("CNN RESULTS")
    print(f"Average Error: {fortyfive_cnn['tot_err'].avg}")
    print(f"Per Class Error: {[(dig, err.avg) for dig, err in enumerate(fortyfive_cnn['pc_err'])]}")


    print("SCALE VARIANCE TESTS")
    print("-"*50)
    print("BASELINE RESULTS")
    print("CAPS RESULTS")
    print(f"Average Error: {zero_caps['tot_err'].avg}")
    print(f"Per Class Error: {[(dig, err.avg) for dig, err in enumerate(zero_caps['pc_err'])]}")
    print("CNN RESULTS")
    print(f"Average Error: {zero_cnn['tot_err'].avg}")
    print(f"Per Class Error: {[(dig, err.avg) for dig, err in enumerate(zero_cnn['pc_err'])]}")
    print()
    print("-"*50)
    print("TWENTY-FIVE PERCENT SCALE RESULTS")
    print("CAPS RESULTS")
    twentyfive_caps, twentyfive_cnn = run_test(capsmodel, cnn, twentyfive_scale)
    print(f"Average Error: {twentyfive_caps['tot_err'].avg}")
    print(f"Per Class Error: {[(dig, err.avg) for dig, err in enumerate(twentyfive_caps['pc_err'])]}")
    print("CNN RESULTS")
    print(f"Average Error: {twentyfive_cnn['tot_err'].avg}")
    print(f"Per Class Error: {[(dig, err.avg) for dig, err in enumerate(twentyfive_cnn['pc_err'])]}")
    print()
    print("-"*50)
    print("FIFTY PERCENT SCALE RESULTS")
    print("CAPS RESULTS")
    fifty_caps, fifty_cnn = run_test(capsmodel, cnn, fifty_scale)
    print(f"Average Error: {fifty_caps['tot_err'].avg}")
    print(f"Per Class Error: {[(dig, err.avg) for dig, err in enumerate(fifty_caps['pc_err'])]}")
    print("CNN RESULTS")
    print(f"Average Error: {fifty_cnn['tot_err'].avg}")
    print(f"Per Class Error: {[(dig, err.avg) for dig, err in enumerate(fifty_cnn['pc_err'])]}")
    print()
    print("-"*50)



    # PLOTTING ROTATIONAL VARIATIONS EFFECT ON OVERALL COUNTING ERROR
    import matplotlib.pyplot as plt
    cats = ["Baseline", "± Fifteen Degrees Rotational Variation", "± Thirty Degrees Rotational Variation", "± Forty-Five Degrees Rotational Variation"]

    caps = [zero_caps['tot_err'].avg, fifteen_caps['tot_err'].avg, thirty_caps['tot_err'].avg, fortyfive_caps['tot_err'].avg]
    cnns = [zero_cnn['tot_err'].avg, fifteen_cnn['tot_err'].avg, thirty_cnn['tot_err'].avg, fortyfive_cnn['tot_err'].avg]

    xpos1 = br1 = np.arange(4) 
    xpos2 = [x + 0.3 for x in br1] 
    
    plt.xlabel('Network Architecture', fontweight='bold')
    plt.xticks([r + 0.3/2 for r in range(4)], cats)

    plt.bar(xpos1, caps, width = 0.3, label="CapsNet")
    plt.bar(xpos2, cnns, width = 0.3, label="CNN")

    plt.ylabel('Mean Average Counting Error')
    plt.title('Effect of Rotational Variation on Counting Error in CapsNet and CNN')
    plt.legend()

    plt.show()

    # PLOTTING SCALE VARIATIONS EFFECT ON OVERALL COUNTING ERROR
    cats = ["Baseline", "± Twenty-Five Percent Scale Variation", "± Fifty Percent Scale Variation"]

    caps = [zero_caps['tot_err'].avg, twentyfive_caps['tot_err'].avg, fifty_caps['tot_err'].avg]
    cnns = [zero_cnn['tot_err'].avg, twentyfive_cnn['tot_err'].avg, fifty_cnn['tot_err'].avg]

    xpos1 = br1 = np.arange(3)
    xpos2 = [x + 0.3 for x in br1]

    plt.xlabel('Network Architecture', fontweight='bold')
    plt.xticks([r + 0.3/2 for r in range(3)], cats)
    plt.bar(xpos1, caps, width = 0.3, label="CapsNet")
    plt.bar(xpos2, cnns, width = 0.3, label="CNN")

    plt.ylabel('Mean Average Counting Error')
    plt.title('Effect of Scale Variation on Counting Error in CapsNet and CNN')
    plt.legend()
    plt.show()


    # PLOTTING THE PER-CLASS EFFECT OF ROTATIONAL VARIATION ON COUNTING ERROR
    # import matplotlib.pyplot as plt
    # cats = ["± Fifteen Degrees Rotational Variation", "± Thirty Degrees Rotational Variation", "± Forty-Five Degrees Rotational Variation"]

    # caps = [fifteen_caps['pc_err'], thirty_caps['pc_err'], fortyfive_caps['pc_err']]
    # cnns = [fifteen_cnn['pc_err'], thirty_cnn['pc_err'], fortyfive_cnn['pc_err']]

    # caps_dig = []
    # cnn_dig = []
    # for i in range(10):
    #     caps_dig.append([caps[j][i].avg for j in range(3)])
    #     cnn_dig.append([cnns[j][i].avg for j in range(3)])
    #     plt.plot(cats,caps_dig[i], label=f"CapsNet Digit {i}", marker='o')
    #     plt.plot(cats,cnn_dig[i], label=f"CNN Digit {i}", marker='x')
        
    
    # plt.legend()
    # plt.grid()
    # plt.show()

            

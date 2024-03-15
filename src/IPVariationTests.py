from torchdataset import *
from convert_to_npz import show_density_map
from capsnet import CapsNetBasic, SegCaps, CCCaps, SegCapsOld, ReconstructionNet, CapsNetComplex
from pilot_utils import TorchUNetModel, TorchUNetModel2
import torch
from capstrain import AverageMeter, compute_acc
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_test(capsmodel, cnn, dataset, show=False):
    caps_results = {
        "pc_err": [AverageMeter() for i in range(10)],
        "tot_err": AverageMeter(),
        "vals": [],
    }
    cnn_results = {
        "pc_err": [AverageMeter() for i in range(10)],
        "tot_err": AverageMeter(),
        "vals": [],
    }

    print("Running Test")
    count = 0
    for i in tqdm(range(1000)):
        input_dict = next(dataset)
        img = input_dict['image'].to(device)
        ground_truth = input_dict["dmap"]
        capsout = capsmodel(img.unsqueeze(0).float().to(device))[0]
        cnnout = cnn(img.unsqueeze(0).float().to(device))[0]
        count += sum(sum(sum(sum(ground_truth.cpu().detach().numpy()))))

        caps_results["tot_err"].update(abs(sum(sum(sum(ground_truth.cpu().detach().numpy()[0]))) / 100 - sum(sum(sum(capsout.cpu().detach().numpy()[0]))) / 100))
        caps_results["vals"].append(abs(sum(sum(sum(ground_truth.cpu().detach().numpy()[0]))) / 100 - sum(sum(sum(capsout.cpu().detach().numpy()[0]))) / 100))

        cnn_results["tot_err"].update(abs(sum(sum(sum(ground_truth.cpu().detach().numpy()[0]))) / 100 - sum(sum(sum(cnnout.cpu().detach().numpy()[0]))) / 100))
        cnn_results["vals"].append(abs(sum(sum(sum(ground_truth.cpu().detach().numpy()[0]))) / 100 - sum(sum(sum(cnnout.cpu().detach().numpy()[0]))) / 100))

        for digit_class in range(10):
            caps_results["pc_err"][digit_class].update(compute_acc(capsout.cpu().detach().numpy(), ground_truth.cpu().detach().numpy(), digit_class))
            cnn_results["pc_err"][digit_class].update(compute_acc(cnnout.cpu().detach().numpy(), ground_truth.cpu().detach().numpy(), digit_class))

        if show and i==0:
            show_density_map(img.cpu().detach().numpy()[0], img.cpu().detach().numpy()[0])

    print(f"Mean Count in this dataset: {count /100 /1000}")


    return caps_results, cnn_results    

def get_95CIs(std):
    n = 1000
    z = 1.96
    return z * (std / np.sqrt(n))

if "__main__" == __name__:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    CAPSWEIGHTS='segcaps_5by5routing.pkl'
    BASELINEWEIGHTS='UNET_WEIGHTS_BEST.pkl'

    rot = False
    scale = False
    noise = True

    baseline = iter(DataLoader(SpreadMNISTDataset(int(1000), train=False, path="baseline"), batch_size=1, shuffle=False))

    fifteen_degree = iter(DataLoader(SpreadMNISTDataset(int(1000), train=False, path="fifteen_degrees"), batch_size=1, shuffle=False))
    thirty_degree = iter(DataLoader(SpreadMNISTDataset(int(1000), train=False, path="thirty_degrees"), batch_size=1, shuffle=False))
    forty_five_degree = iter(DataLoader(SpreadMNISTDataset(int(1000), train=False, path="fortyfive_degrees"), batch_size=1, shuffle=False))
    ninety_degree = iter(DataLoader(SpreadMNISTDataset(int(1000), train=False, path="ninety_degrees"), batch_size=1, shuffle=False))
    twentyfive_scale = iter(DataLoader(SpreadMNISTDataset(int(1000), train=False, path="twentyfive_perc"), batch_size=1, shuffle=False))
    fifty_scale = iter(DataLoader(SpreadMNISTDataset(int(1000), train=False, path="fifty_perc"), batch_size=1, shuffle=False))
    severty_five_scale = iter(DataLoader(SpreadMNISTDataset(int(1000), train=False, path="seventyfive_perc"), batch_size=1, shuffle=False))

    capsmodel = CapsNetComplex(10).to(device)
    cnn = TorchUNetModel(10).to(device)

    capsmodel.load_state_dict(torch.load(CAPSWEIGHTS))
    cnn.load_state_dict(torch.load(BASELINEWEIGHTS))

    zero_caps, zero_cnn = run_test(capsmodel, cnn, baseline)

    if rot:
        print("ROTATIONAL VARIANCE TESTS")
        print("-"*50)
        
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




    # print("SCALE VARIANCE TESTS")
    if scale:
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
        print("SEVENTY-FIVE PERCENT SCALE RESULTS")
        seventyfive_caps, seventyfive_cnn = run_test(capsmodel, cnn, severty_five_scale)
        print(f"Average Error: {seventyfive_caps['tot_err'].avg}")
        print(f"Per Class Error: {[(dig, err.avg) for dig, err in enumerate(seventyfive_caps['pc_err'])]}")
        print("CNN RESULTS")
        print(f"Average Error: {seventyfive_cnn['tot_err'].avg}")
        print(f"Per Class Error: {[(dig, err.avg) for dig, err in enumerate(seventyfive_cnn['pc_err'])]}")



    # # # PLOTTING ROTATIONAL VARIATIONS EFFECT ON OVERALL COUNTING ERROR
    import matplotlib.pyplot as plt
    #mpl stuff
    plt.rc('xtick', labelsize=20) 
    plt.rc('ytick', labelsize=20)  
    if rot:
        cats = ["Baseline", "± 15 Degrees Variation", "± 30 Degrees Variation", "± 45 Degrees Variation"]

        caps = [zero_caps['tot_err'].avg, fifteen_caps['tot_err'].avg, thirty_caps['tot_err'].avg, fortyfive_caps['tot_err'].avg]
        caps_cis = [get_95CIs(np.std(zero_caps["vals"])), get_95CIs(np.std(fifteen_caps["vals"])), get_95CIs(np.std(thirty_caps["vals"])), get_95CIs(np.std(fortyfive_caps["vals"]))]

        cnns = [zero_cnn['tot_err'].avg, fifteen_cnn['tot_err'].avg, thirty_cnn['tot_err'].avg, fortyfive_cnn['tot_err'].avg]
        cnn_cis = [get_95CIs(np.std(zero_cnn["vals"])), get_95CIs(np.std(fifteen_cnn["vals"])), get_95CIs(np.std(thirty_cnn["vals"])), get_95CIs(np.std(fortyfive_cnn["vals"]))]

        xpos1 = np.arange(4) 
        xpos2 = [x + 0.3 for x in xpos1] 
        
        plt.xlabel('Network Architecture', fontweight='bold', fontsize=20)
        plt.xticks([r + 0.3/2 for r in range(4)], cats)

        plt.bar(xpos1, caps, width = 0.3, label="CapsNet", yerr=caps_cis, capsize=5)
        plt.bar(xpos2, cnns, width = 0.3, label="CNN", yerr=cnn_cis, capsize=5)

        plt.ylabel('Mean Average Counting Error', fontweight='bold', fontsize=20)
        # plt.title('Effect of Rotational Variation on Counting Error in CapsNet and CNN')
        plt.legend(fontsize=20)

        plt.show()

    # # # PLOTTING SCALE VARIATIONS EFFECT ON OVERALL COUNTING ERROR
    if scale:
        cats = ["Baseline", "± 25% Variation", "± 50% Variation", "± 75% Variation"]   

        caps = [zero_caps['tot_err'].avg, twentyfive_caps['tot_err'].avg, fifty_caps['tot_err'].avg, seventyfive_caps['tot_err'].avg]
        caps_cis = [get_95CIs(np.std(zero_caps["vals"])), get_95CIs(np.std(twentyfive_caps["vals"])), get_95CIs(np.std(fifty_caps["vals"])), get_95CIs(np.std(seventyfive_caps["vals"]))]
        cnns = [zero_cnn['tot_err'].avg, twentyfive_cnn['tot_err'].avg, fifty_cnn['tot_err'].avg, seventyfive_cnn['tot_err'].avg]
        cnn_cis = [get_95CIs(np.std(zero_cnn["vals"])), get_95CIs(np.std(twentyfive_cnn["vals"])), get_95CIs(np.std(fifty_cnn["vals"])), get_95CIs(np.std(seventyfive_cnn["vals"]))]

        xpos1 = br1 = np.arange(4)
        xpos2 = [x + 0.3 for x in br1]

        plt.xlabel('Network Architecture', fontweight='bold', fontsize=20)
        plt.xticks([r + 0.3/2 for r in range(4)], cats)
        plt.bar(xpos1, caps, width = 0.3, label="CapsNet", yerr=caps_cis, capsize=5)
        plt.bar(xpos2, cnns, width = 0.3, label="CNN", yerr=cnn_cis, capsize=5)

        plt.ylabel('Mean Average Counting Error', fontweight='bold', fontsize=20)
        # plt.title('Effect of Scale Variation on Counting Error in CapsNet and CNN')
        plt.legend(fontsize=20)
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

    # NOISE ERROR
    if noise:
        db = range(0, 11)
        delta_db = [3*d for d in db]
        capsnt = []
        cnnnt = []
        std = 2.55
        for noise_level in db:
            baseline = iter(DataLoader(SpreadMNISTDataset(int(1000), train=False, noise_db=noise_level, std=2.55), batch_size=1, shuffle=True))
            capsout, cnnout = run_test(capsmodel, cnn, baseline)
            print(f"RESULTS FOR NOISE LEVEL {noise_level}dB")
            print(f"Caps Average Error: {capsout['tot_err'].avg}")
            capsnt.append(capsout['tot_err'].avg)
            print(f"Per Class Error: {[(dig, err.avg) for dig, err in enumerate(capsout['pc_err'])]}")
            print()
            print(f"CNN Average Error: {cnnout['tot_err'].avg}")
            cnnnt.append(cnnout['tot_err'].avg)
            print(f"Per Class Error: {[(dig, err.avg) for dig, err in enumerate(cnnout['pc_err'])]}")

                
        plt.plot(delta_db,capsnt, label="CapsNet", marker='o')
        plt.plot(delta_db,cnnnt, label="CNN", marker='x')
        plt.xlabel('Noise Level (Change in dB)')
        plt.ylabel('Mean Average Counting Error')
        # plt.title('Effect of Noise on Counting Error in CapsNet and CNN')
        plt.grid()
        plt.legend()
        plt.show()
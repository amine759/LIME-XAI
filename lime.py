"""
a from-scratch implementation of the LIME algorithm for explainable AI, applied to an LSTM classifier trained on Arabic data. 
While LIME is model-agnostic and can be adapted to various use cases, there may be minor differences in implementation, particularly
in the perturbation generation steps, between NLP settings and tabular data.
"""

import numpy as np
import torch
from model import LSTMClassifier
from sklearn.metrics.pairwise import cosine_distances
from sklearn.linear_model import Ridge, Lasso


embedding_dim = 512
best_config = {
    "n_layers": 1,
    "n_hidden": 95,
    "lr": 0.012942851972629671,
}  # obtained from optuna trials
num_classes = 3

loaded_model = LSTMClassifier(
    embedding_dim, best_config["n_hidden"], best_config["n_layers"], num_classes
).cpu()

loaded_model.load_state_dict(torch.load("LstmClassifier.pt"))
loaded_model.eval()
"""out : 
LSTMClassifier(
  (lstm): LSTM(512, 95, batch_first=True)
  (lstm_layers): ModuleList()
  (fc): Linear(in_features=95, out_features=3, bias=True)
  (softmax): Softmax(dim=1)
)
"""
"""
steps for the lime implementation : 
1.  Generate Perturbations: Create perturbed versions of the instance we want to explain.
2.  Get Predictions: Use the original model to get predictions for each perturbed instance.
3.  Weight Perturbations: Calculate weights based on similarity to the original instance.
4.  Fit Local Model: Fit a linear model (lasso regression) to the weighted perturbations.
5. Interpret the Model: Extract and interpret the coefficients of the linear model.
"""

# Example usage:
text_instance = "يتأثر الاقتصاد والسياسة والصحة ببعضها البعض بشكل كبير، حيث تُحدد السياسات الاقتصادية القرارات الصحية وتؤثر على توافر الخدمات الطبية وتمويلها، مما يعزز التنمية والرفاهية الاجتماعية بشكل متكامل ومستدام"
num_samples = len(text_instance)


# step 1
def perturb_text(text, num_samples=num_samples, mask_string="MASK"):
    words = text.split()
    perturbations = []

    for i in range(len(words)):
        for _ in range(num_samples // len(words)):
            perturbed_text = words[:]
            perturbed_text[i] = mask_string
            perturbations.append(" ".join(perturbed_text))
    return perturbations


# step 2
def get_predictions(perturbations, pipeline):
    # pipeline.predict_proba([text])
    probabilities = []
    for sample in perturbations:
        probabilities.append(pipeline.predict_proba([sample]).flatten())
    return np.array(probabilities)


# step 3
def calculate_weights(original_text, perturbations, kernel_width=25):
    original_vector = np.ones((1, len(original_text.split())))
    perturbed_vectors = np.array(
        [
            [1 if word != "MASK" else 0 for word in text.split()]
            for text in perturbations
        ]
    )
    print(original_vector.shape, perturbed_vectors.shape)

    distances = cosine_distances(perturbed_vectors, original_vector).flatten()
    weights = np.sqrt(np.exp(-(distances**2) / kernel_width**2))
    return weights


# Step 4: Fit Local Model
"""
in the paper Lasso is used whereas in the lime_text library they have used ridge by default, we'll investigate both of em
The final interpretable model is a sparse linear model  (Regularized), as lasso tends to discriminate unimportant features 
whereas Ridge just reduce the complexity of the model, for textual data the features for the linear models are the tokens.
So the attribution scores are the coefficient of the sparse linear model.
"""


def fit_local_model(perturbations, predictions, weights, estimator):
    perturbed_vectors = np.array(
        [
            [1 if word != "MASK" else 0 for word in text.split()]
            for text in perturbations
        ]
    )
    models = []
    num_classes = 3
    for class_idx in range(num_classes):
        if estimator == "lasso":
            model = Lasso(alpha=0.0001)
        else:
            model = Ridge()
        model.fit(perturbed_vectors, predictions[:, class_idx], sample_weight=weights)
        models.append(model)
    return models


# step 5 : interpret model
def interpret_local_model(models, feature_names):
    feature_importances = {}
    for class_idx, model in enumerate(models):
        coefficients = model.coef_
        feature_importance = sorted(
            zip(feature_names, coefficients), key=lambda x: x[1], reverse=True
        )
        feature_importances[class_idx] = feature_importance
    return feature_importances


# putting it all together :
def lime_explain_instance(
    text_instance,
    classifier_fn,
    local_model_name,
    num_samples=num_samples,
    kernel_width=25,
):
    perturbations = perturb_text(text_instance, num_samples)
    predictions = get_predictions(perturbations, classifier_fn)
    weights = calculate_weights(text_instance, perturbations, kernel_width)

    local_model = fit_local_model(perturbations, predictions, weights, local_model_name)

    feature_names = text_instance.split()
    feature_importances = interpret_local_model(local_model, feature_names)
    return feature_importances


explanation_ridge = lime_explain_instance(
    text_instance, loaded_model, local_model_name="ridge"
)
explanation_lasso = lime_explain_instance(
    text_instance, loaded_model, local_model_name="lasso"
)

'''
label mappings :
{'اقتصاد': 0, 'سياسة': 1, 'صحة': 2}
'''

print(explanation_lasso)
"""
out : 
{0: [('الاقتصادية', 0.1539086660197635),
  ('الاقتصاد', 0.1364346059642764),
  ('وتمويلها،', 0.055520772234569964),
  ('التنمية', 0.016561121741366425),
  ('القرارات', 0.014579378723562807),
  ('كبير،', 0.010199872667873455),
  ('ببعضها', 0.0075174527982887875),
  ('تُحدد', 0.004319815114932761),
  ('يعزز', 0.0021468333389214342),
  ('البعض', 0.0018990408459094643),
  ('بشكل', 0.0),
  ('حيث', 0.0),
  ('وتؤثر', 0.0),
  ('بشكل', 0.0),
  ('متكامل', -0.0),
  ('ومستدام', -0.0),
  ('على', -0.0007014796442538343),
  ('مما', -0.0026148089235696996),
  ('الخدمات', -0.004903256907064974),
  ('يتأثر', -0.009787306285464792),
  ('توافر', -0.023225128929203465),
  ('الاجتماعية', -0.04867371708792062),
  ('والرفاهية', -0.053732593457436605),
  ('والسياسة', -0.0721500841896599),
  ('الطبية', -0.0813212910356521),
  ('الصحية', -0.08503994999950634),
  ('السياسات', -0.08888202680990114),
  ('والصحة', -0.0889273397458268)],
 1: [('السياسات', 0.054113358705253196),
  ('والسياسة', 0.05177301244487891),
  ('الاجتماعية', 0.039541205938145384),
  ('والصحة', 0.01876351986096369),
  ('والرفاهية', 0.018309028276998388),
  ('الصحية', 0.014378258757711563),
  ('الطبية', 0.012631469971363047),
  ('التنمية', 0.010427155855036145),
  ('ومستدام', 0.005936232438350716),
  ('يتأثر', -0.0),
  ('البعض', 0.0),
  ('بشكل', -0.0),
  ('كبير،', -0.0),
  ('حيث', -0.0),
  ('تُحدد', -0.0),
  ('وتؤثر', -0.0),
  ('على', 0.0),
  ('توافر', 0.0),
  ('مما', -0.0),
  ('بشكل', -0.0),
  ('متكامل', -0.0),
  ('يعزز', -0.0007553655799410703),
  ('ببعضها', -0.0009931648701299396),
  ('القرارات', -0.005062265618343956),
  ('الخدمات', -0.013100197585677681),
  ('وتمويلها،', -0.016978582476705972),
  ('الاقتصاد', -0.05053039897439338),
  ('الاقتصادية', -0.054363911329429665)],
 2: [('الصحية', 0.0674514332790254),
  ('والصحة', 0.0669551145556204),
  ('الطبية', 0.06547891987875855),
  ('والرفاهية', 0.03221178320698775),
  ('السياسات', 0.031558939660029525),
  ('توافر', 0.02177136716304041),
  ('الخدمات', 0.020392776489335917),
  ('والسياسة', 0.017168504915488627),
  ('يتأثر', 0.010592295919373274),
  ('الاجتماعية', 0.00592065173333997),
  ('مما', 0.003895345869011919),
  ('على', 0.00011584737845922097),
  ('بشكل', -0.0),
  ('حيث', -0.0),
  ('وتؤثر', 0.0),
  ('يعزز', -0.0),
  ('بشكل', -0.0),
  ('متكامل', 0.0),
  ('البعض', -0.0032682771712963703),
  ('ببعضها', -0.004133117031677499),
  ('تُحدد', -0.004530740765918668),
  ('ومستدام', -0.006335728890731966),
  ('القرارات', -0.007127059537992428),
  ('كبير،', -0.009843397576897733),
  ('التنمية', -0.030199988537495788),
  ('وتمويلها،', -0.036153235189815364),
  ('الاقتصاد', -0.08351255661122446),
  ('الاقتصادية', -0.09715471329808095)]}

"""
print(explanation_ridge)
"""
out : 
{0: [('الاقتصادية', 0.1420186693301159),
  ('الاقتصاد', 0.1267258465120105),
  ('وتمويلها،', 0.05593150486235373),
  ('التنمية', 0.021842670849024495),
  ('القرارات', 0.02010588841421111),
  ('كبير،', 0.016272408359375212),
  ('ببعضها', 0.013924536774424895),
  ('تُحدد', 0.011127723019603575),
  ('يعزز', 0.00922988883983643),
  ('البعض', 0.009008807768962464),
  ('بشكل', 0.006995556648915985),
  ('حيث', 0.006707040375248894),
  ('وتؤثر', 0.006213297865703278),
  ('بشكل', 0.006071282353556588),
  ('ومستدام', 0.00489155746014559),
  ('متكامل', 0.003506814947106014),
  ('على', 0.0018357987850861953),
  ('مما', 0.0001631658471293497),
  ('الخدمات', -0.0018401238469824895),
  ('يتأثر', -0.006118738825128025),
  ('توافر', -0.017872074038625556),
  ('الاجتماعية', -0.040137269162227346),
  ('والرفاهية', -0.04456405383189276),
  ('والسياسة', -0.060685343932572945),
  ('الطبية', -0.06870559566151802),
  ('الصحية', -0.07196068716339522),
  ('السياسات', -0.07532352895836646),
  ('والصحة', -0.07536504359210462)],
 1: [('السياسات', 0.0471718908919669),
  ('والسياسة', 0.04512430121121747),
  ('الاجتماعية', 0.034420793049439534),
  ('والصحة', 0.01624092016952709),
  ('والرفاهية', 0.01584267174839145),
  ('الصحية', 0.012403505742602423),
  ('الطبية', 0.010874967960033835),
  ('التنمية', 0.00894606951679905),
  ('ومستدام', 0.005016410851482964),
  ('توافر', -0.0017147293572468197),
  ('البعض', -0.0017871061574603988),
  ('على', -0.0024741055455442316),
  ('بشكل', -0.002724229909964548),
  ('تُحدد', -0.002801131075098698),
  ('حيث', -0.0028213342550538296),
  ('بشكل', -0.0032718932008618576),
  ('كبير،', -0.003297318306323302),
  ('متكامل', -0.0034307740329646346),
  ('يتأثر', -0.0036883890246093656),
  ('مما', -0.00410781199511383),
  ('وتؤثر', -0.00432002035610754),
  ('يعزز', -0.005738597817184272),
  ('ببعضها', -0.005946249216968365),
  ('القرارات', -0.009506898332429542),
  ('الخدمات', -0.016540193102166256),
  ('وتمويلها،', -0.019933870986581666),
  ('الاقتصاد', -0.04929109939279418),
  ('الاقتصادية', -0.05264577907698631)],
 2: [('الصحية', 0.059557119633362667),
  ('والصحة', 0.05912411786374602),
  ('الطبية', 0.05783061154885872),
  ('والرفاهية', 0.028721356151989164),
  ('السياسات', 0.02815164147154751),
  ('توافر', 0.019586777464360227),
  ('الخدمات', 0.018380372508359236),
  ('والسياسة', 0.01556101190039993),
  ('يتأثر', 0.009807193187834582),
  ('الاجتماعية', 0.005716495816080353),
  ('مما', 0.003944724524597305),
  ('على', 0.0006383395022661833),
  ('متكامل', -7.60146915910393e-05),
  ('وتؤثر', -0.0018932252100141596),
  ('بشكل', -0.0033470392595573275),
  ('يعزز', -0.003491323473422116),
  ('بشكل', -0.0037236763410506547),
  ('حيث', -0.003885758128738459),
  ('البعض', -0.007221688427467353),
  ('ببعضها', -0.007978254815648388),
  ('تُحدد', -0.008326670030079511),
  ('ومستدام', -0.009907955127593836),
  ('القرارات', -0.010598911705168754),
  ('كبير،', -0.012975083388275003),
  ('التنمية', -0.030788818451398185),
  ('وتمويلها،', -0.035997653288026406),
  ('الاقتصاد', -0.07743473393518174),
  ('الاقتصادية', -0.08937295530018852)]}
"""
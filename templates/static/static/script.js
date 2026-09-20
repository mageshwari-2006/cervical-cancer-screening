const imageInput = document.getElementById("imageInput");
const analyzeBtn = document.getElementById("analyzeBtn");

const preview = document.getElementById("preview");
const resultText = document.getElementById("resultText");
const confidenceText = document.getElementById("confidenceText");
const qualityText = document.getElementById("qualityText");
const lesionText = document.getElementById("lesionText");


imageInput.addEventListener("change", function () {

    const file = imageInput.files[0];

    if (file) {

        const imageURL = URL.createObjectURL(file);

        preview.innerHTML = `
            <img src="${imageURL}" alt="Uploaded Colposcopy Image">
        `;

        resultText.textContent =
            "Image uploaded successfully. Click Analyze Image.";

        confidenceText.textContent = "Not available";
        qualityText.textContent = "Waiting for analysis";
        lesionText.textContent =
            "Not available from the current model.";
    }
});


analyzeBtn.addEventListener("click", async function () {

    const file = imageInput.files[0];

    if (!file) {

        resultText.textContent =
            "Please upload an image first.";

        return;
    }


    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "Analyzing...";

    resultText.textContent =
        "AI model is analyzing the image...";

    confidenceText.textContent = "Calculating...";


    const formData = new FormData();

    formData.append("image", file);


    try {

        const response = await fetch("/predict", {

            method: "POST",

            body: formData

        });


        const data = await response.json();


        if (!response.ok) {

            throw new Error(
                data.error || "Analysis failed"
            );

        }


        const normal =
            (data.normal_probability * 100).toFixed(2);

        const abnormal =
            (data.abnormal_probability * 100).toFixed(2);


        resultText.innerHTML = `
            <strong>${data.prediction}</strong>
            <br><br>
            Normal Probability: ${normal}%
            <br>
            Abnormal Probability: ${abnormal}%
        `;


        // Confidence = higher probability
        const confidence =
            Math.max(
                data.normal_probability,
                data.abnormal_probability
            ) * 100;


        confidenceText.textContent =
            confidence.toFixed(2) + "%";


        // Current backend does not perform separate quality analysis
        qualityText.textContent =
            "Image preprocessing completed.";


        // Current model does not provide lesion localization
        lesionText.textContent =
            "Lesion localization is not provided by the current model.";


    } catch (error) {

        console.error(error);

        resultText.textContent =
            "Analysis failed: " + error.message;

        confidenceText.textContent =
            "Not available";

    }


    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze Image";

});

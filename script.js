function selectModel() {
    var modelSelect = document.getElementById("modelSelect");
    var selectedModel = modelSelect.options[modelSelect.selectedIndex].value;

    if (selectedModel === "Age estimation model") {
        document.getElementById("modelDetails").innerHTML = "Age estimation model details...";
    } else if (selectedModel === "Sex estimation model") {
        document.getElementById("modelDetails").innerHTML = "Sex estimation model details...";
    } else {
        document.getElementById("modelDetails").innerHTML = "";
    }
}

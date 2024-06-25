function updateDefaults() {
    var modelSelect = document.getElementById("model_select").value;
    console.log("Dropdown value changed: ", modelSelect);

    if (modelSelect == "0") {
        document.getElementById("frompredict").value = "Age";
        document.getElementById("node0input").value = "Younger";
        document.getElementById("node1input").value = "Older";
        console.log("Age model selected");
    } else if (modelSelect == "1") {
        document.getElementById("frompredict").value = "Gender";
        document.getElementById("node0input").value = "Male";
        document.getElementById("node1input").value = "Female";
        console.log("Gender model selected");
    } else {
        document.getElementById("frompredict").value = "";
        document.getElementById("node0input").value = "";
        document.getElementById("node1input").value = "";
        console.log("No model selected");
    }

    console.log("frompredict: ", document.getElementById("frompredict").value);
    console.log("node0input: ", document.getElementById("node0input").value);
    console.log("node1input: ", document.getElementById("node1input").value);
}

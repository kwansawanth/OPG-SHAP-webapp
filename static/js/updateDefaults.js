function updateDefaults() {
    var modelSelect = document.getElementById("model_select").value;
    if (modelSelect == "0") { // Age estimation model
        document.getElementById("frompredict").value = "Age";
        document.getElementById("node0input").value = "Younger";
        document.getElementById("node1input").value = "Older";
    } else if (modelSelect == "1") { // Sex estimation model
        document.getElementById("frompredict").value = "Gender";
        document.getElementById("node0input").value = "Male";
        document.getElementById("node1input").value = "Female";
    } else { // models import from users
        document.getElementById("frompredict").value = "";
        document.getElementById("node0input").value = "";
        document.getElementById("node1input").value = "";
    }
}

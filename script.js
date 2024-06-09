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
const dropArea = document.querySelector(".drop_box1"),
  button = dropArea.querySelector("button"),
  dragText = dropArea.querySelector("header"),
  input = dropArea.querySelector("input");
let file;
var filename;

button.onclick = () => {
  input.click();
};

input.addEventListener("change", function (e) {
  var fileName = e.target.files[0].name;
  let filedata = `
    <form action="" method="post">
    <div class="form">
    <h4>${fileName}</h4>
    <input type="email" placeholder="Enter email upload file">
    <button class="btn">Upload</button>
    </div>
    </form>`;
  dropArea.innerHTML = filedata;
});
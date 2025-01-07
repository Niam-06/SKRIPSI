document.addEventListener("DOMContentLoaded", function() {
    var fileInput = document.getElementById('file');
    var fileLabel = document.getElementById('file-label');
    
    fileInput.addEventListener('change', function(event) {
        var fileName = fileInput.files[0].name;
        fileLabel.textContent = fileName ? fileName : "No file chosen";
    });
});

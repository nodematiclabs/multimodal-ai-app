document.getElementById('imageForm').addEventListener('submit', upload);
document.getElementById('descriptionForm').addEventListener('submit', upload);
document.getElementById('cluster').addEventListener('click', cluster);

function upload(event) {
    event.preventDefault();

    var formData = new FormData(this);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        return response.json();
    })
    .then(data => {
        if (data.image) {
            let img = document.createElement('img');
            img.src = data.image;
            img.alt = 'Uploaded image';

            let imageContainer = document.getElementById('imageContainer');
            imageContainer.appendChild(img);
        } else if (data.text) {
            let text = document.createElement('p');
            text.textContent = data.text;

            let imageContainer = document.getElementById('textContainer');
            imageContainer.appendChild(text)
        }
        event.target.reset();
    })
    .catch(error => {
        console.error(error);
    });
};

function cluster() {
    fetch('/cluster', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            k: document.getElementById('k').value
        }),
    })
    .then(response => response.json())
    .then(data => {
        colors = [
            'lightblue',
            'lightcoral',
            'lightcyan',
            'lightgoldenrodyellow',
            'lightgrey',
            'lightgreen',
            'lightpink',
            'lightsalmon',
            'lightseagreen',
            'lightskyblue',
            'lightsteelblue'
        ]
        data.vectors.forEach(asset => {
            if (asset.image != "") {
                document.querySelector(`img[src="/uploads/${asset.image}"]`).style.backgroundColor = colors[asset.cluster];
            } else if (asset.description != "") {
                // Get the p element with text that matches the asset.description
                document.querySelectorAll('p').forEach(p => {
                    if (p.textContent == asset.description) {
                        p.style.backgroundColor = colors[asset.cluster];
                    }
                })
            }
        })
    })
    .catch(error => console.error(error));
}
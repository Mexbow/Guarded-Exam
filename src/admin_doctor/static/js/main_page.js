document.addEventListener('DOMContentLoaded', function() {
    let animatedTitle = document.querySelector("#animatedTitle");
    const text = "OXFORD";
    const speed = 300;
    const delay = 850;
    let i = 0;

    function typeWriter() {
        if (i < text.length) {
            if (animatedTitle.children[i]) {
                animatedTitle.children[i].style.visibility = "visible";
            }
            i++;
            setTimeout(typeWriter, speed);
        } else {
            setTimeout(() => {
                Array.from(animatedTitle.children).forEach(span => {
                    span.style.visibility = "hidden";
                });
                i = 0;
                setTimeout(typeWriter, delay);
            }, delay);
        }
    }

    if (animatedTitle && animatedTitle.children.length > 0) {
        typeWriter();
    } else {
        console.error("animatedTitle element not found or has no children");
    }
});
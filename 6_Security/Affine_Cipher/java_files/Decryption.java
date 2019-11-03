public class Decryption {
    public String Decryption(int a, int a_inverse, int b, int L4PN, String ciphertext) {
        String plaintext = "";
        String[] Int_ciph = Ciphertext.split("");
        for (int i = 0; i < int_ciph.length; i++) {
            int temp = ((Integer.parseInt(Int_ciph[i]) - b) * a_inverse) % LP4N;
            if (temp < 0) {
                temp = temp + L4PN;
            }
            plaintext = plaintext + temp + "";
        }
        return plaintext;
    }

    public void Example() {
        System.out.println("Hello World!");
    }
}
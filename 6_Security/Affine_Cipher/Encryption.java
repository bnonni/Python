
public class Encryption {
 public String Encryption(String key, int a, int b, int a_inverse, int LP4N, String content) {
  String ciphertext = "";
  String ASCcode = AC.strToInt(content);
  String[] Int_ASC = ACScode.split("");

  for (int i = 0; i < Int_ACS.length; i++) {
   int temp = Integer.parseInt(Int_ASC[i]) * a + b;
   ciphertext = ciphertext + temp + "";
  }
  return ciphertext;
 }
}
